# Tests for the semantic index (R1) and the retrieval step with reranking (R2), on a fake embedding model.

import json

import pytest

from sensai.core.config import DEFAULTS
from sensai.knowledge.rag import SemanticIndex, cosine
from sensai.knowledge.retrieval import Retriever, profile_level
from sensai.llm.ollama import OllamaError

FEATURES = ("topic", "subject", "te-form", "counter", "honorific")
CORPUS = """# Particle は: the topic [N5]
は marks the topic of a sentence.

# Particle が: the subject [N5]
が marks the subject.

# The te-form [N5]
The te-form joins verbs.

# Honorific speech [N3]
Honorific verbs raise the listener.
"""


class EmbeddingClient:
    """Embeds a text as the count of a few keywords, and answers the reranking prompt with fixed scores."""

    def __init__(self, scores=None, embed_error=None, chat_error=None):
        self.scores = scores
        self.embed_error = embed_error
        self.chat_error = chat_error
        self.embedded = []
        self.chats = []

    def embed(self, text, model, **options):
        return self.embed_many([text], model, **options)[0]

    def embed_many(self, texts, model, **options):
        if self.embed_error:
            raise self.embed_error
        self.embedded.extend(texts)
        return [[text.lower().count(word) + 0.01 for word in FEATURES] for text in texts]

    def chat_stream(self, model, messages, **options):
        self.chats.append((model, messages, options))
        if self.chat_error:
            raise self.chat_error
        yield json.dumps({"scores": self.scores}) if isinstance(self.scores, list) else str(self.scores)


def settings(tmp_path, **overrides):
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    (corpus / "grammar.md").write_text(CORPUS, encoding="utf-8")
    return {**DEFAULTS["rag"], "corpus_dir": str(corpus), "index_path": str(tmp_path / "index.json"),
            "min_similarity": 0.5, **overrides}


def test_cosine():
    assert cosine([1, 0], [1, 0]) == 1.0 and cosine([1, 0], [0, 1]) == 0.0 and cosine([0, 0], [1, 1]) == 0.0


def test_semantic_search_ranks_by_similarity_and_filters_by_level(tmp_path):
    index = SemanticIndex(EmbeddingClient(), cache_path=tmp_path / "cache.json")
    index.ingest_markdown(CORPUS, "grammar.md")
    hits = index.search(index.embed_query("which particle marks the topic?"), max_jlpt="N5", top_k=2)
    assert hits[0].section == "Particle は: the topic" and hits[0].score > hits[1].score
    honorific = index.embed_query("honorific")
    assert all(hit.jlpt == "N5" for hit in index.search(honorific, max_jlpt="N5"))
    assert index.search(honorific, max_jlpt="N3")[0].section == "Honorific speech"


def test_embeddings_are_cached_by_content_and_reused(tmp_path):
    client = EmbeddingClient()
    index = SemanticIndex(client, cache_path=tmp_path / "cache.json")
    index.ingest_markdown(CORPUS, "grammar.md")
    assert index.embed_chunks() == 8 and index.embed_chunks() == 0
    again = SemanticIndex(EmbeddingClient(embed_error=OllamaError("connection", "down")),
                          cache_path=tmp_path / "cache.json")
    again.ingest_markdown(CORPUS, "grammar.md")
    assert again.embed_chunks() == 0


def test_a_broken_cache_is_ignored(tmp_path):
    (tmp_path / "cache.json").write_text("{not json", encoding="utf-8")
    index = SemanticIndex(EmbeddingClient(), cache_path=tmp_path / "cache.json")
    index.ingest_markdown(CORPUS, "grammar.md")
    assert index.embed_chunks() == 8


def test_the_model_reranks_and_drops_near_misses(tmp_path):
    client = EmbeddingClient(scores=[{"id": 1, "relevance": 0.2}, {"id": 2, "relevance": 0.9}])
    retrieval = Retriever(settings(tmp_path, min_similarity=0.0, candidate_k=2), client).retrieve(
        "topic or subject?", "Level: N5", "qwen3.5:9b")
    assert retrieval.mode == "embeddings+model" and retrieval.level == "N5"
    assert [chunk.score for chunk in retrieval.kept] == [0.9]
    dropped = [trace for trace in retrieval.scored if trace.dropped]
    assert len(dropped) == 1 and dropped[0].rerank_score == 0.2
    assert client.chats[0][2]["format"] == "json"
    assert retrieval.citations[0] in retrieval.context_note()


def test_an_off_topic_question_gets_no_passage_and_no_citation(tmp_path):
    client = EmbeddingClient(scores=[{"id": 1, "relevance": 0}, {"id": 2, "relevance": 0}])
    retrieval = Retriever(settings(tmp_path, min_similarity=0.0, candidate_k=2), client).retrieve(
        "tell me a joke", None, "qwen3.5:9b")
    assert retrieval.kept == [] and retrieval.context_note() is None and retrieval.citations == []


def test_nothing_similar_enough_skips_the_reranker(tmp_path):
    client = EmbeddingClient(scores=[])
    retrieval = Retriever(settings(tmp_path, min_similarity=0.99), client).retrieve("bake bread", None, "m")
    assert (retrieval.mode, retrieval.kept, client.chats) == ("embeddings", [], [])


def test_a_failing_reranker_falls_back_to_the_hybrid_score(tmp_path):
    for client in (EmbeddingClient(scores="not json"), EmbeddingClient(chat_error=OllamaError("timeout", "slow"))):
        retrieval = Retriever(settings(tmp_path, hybrid_threshold=0.3), client).retrieve(
            "the te-form", None, "qwen3.5:9b")
        assert retrieval.mode == "embeddings+hybrid"
        assert retrieval.kept[0].section == "The te-form"


def test_without_the_embedding_model_retrieval_uses_word_overlap(tmp_path):
    client = EmbeddingClient(embed_error=OllamaError("model_missing", "no bge-m3"))
    retrieval = Retriever(settings(tmp_path, lexical_threshold=0.1), client).retrieve("te-form joins verbs", None)
    assert retrieval.mode == "lexical" and retrieval.kept[0].section == "The te-form"


def test_reranking_can_be_turned_off(tmp_path):
    retrieval = Retriever(settings(tmp_path, rerank=False, top_k=1), EmbeddingClient()).retrieve("topic", None, "m")
    assert len(retrieval.kept) == 1 and retrieval.scored[0].dropped is False


def test_an_empty_corpus_retrieves_nothing(tmp_path):
    empty = settings(tmp_path, corpus_dir=str(tmp_path / "missing"))
    assert Retriever(empty, EmbeddingClient()).retrieve("topic", None, "m").mode == "empty"


def test_the_event_fields_list_kept_and_dropped_chunks(tmp_path):
    client = EmbeddingClient(scores=[{"id": 1, "relevance": 1}, {"id": 2, "relevance": 0.1}])
    fields = Retriever(settings(tmp_path, min_similarity=0.0, candidate_k=2), client).retrieve(
        "topic", None, "m").as_event_fields()
    assert fields["kept"][0]["score"] == 1.0 and fields["dropped"][0]["dropped"] is True


def test_profile_level():
    assert profile_level("Name: Bryan. JLPT n4 learner") == "N4"
    assert profile_level(None, "N5") == "N5"
    assert profile_level("no level here") is None


def test_reranker_scores_on_a_0_to_10_scale_and_zero_based_ids(tmp_path):
    client = EmbeddingClient(scores=[{"id": 0, "relevance": 9}, {"id": 1, "relevance": 3}])
    retrieval = Retriever(settings(tmp_path, min_similarity=0.0, candidate_k=2), client).retrieve("topic", None, "m")
    assert [chunk.score for chunk in retrieval.kept] == [0.9]


def test_profile_level_prefers_the_current_level_over_a_goal():
    assert profile_level("I want to reach N3, currently N5") == "N5"
    assert profile_level("Passed N5 last year; now studying for N4") == "N4"
    assert profile_level("Level: N5. Goal: N3") == "N5"


def test_the_cache_keeps_only_current_chunks(tmp_path):
    cache = tmp_path / "cache.json"
    first = SemanticIndex(EmbeddingClient(), cache_path=cache)
    first.ingest_markdown(CORPUS, "grammar.md")
    first.embed_chunks()
    second = SemanticIndex(EmbeddingClient(), cache_path=cache)
    second.ingest_markdown("# New [N5]\nnew text", "new.md")
    second.embed_chunks()
    assert len(json.loads(cache.read_text(encoding="utf-8"))) == 2


def test_fewer_embeddings_than_texts_is_an_error(tmp_path):
    class Short(EmbeddingClient):
        def embed_many(self, texts, model, **options):
            return super().embed_many(texts, model, **options)[:-1]
    index = SemanticIndex(Short())
    index.ingest_markdown(CORPUS, "grammar.md")
    with pytest.raises(ValueError, match="returned"):
        index.embed_chunks()


def test_a_subheader_inherits_its_section_level():
    from sensai.knowledge.rag import BasicRAGIndex
    index = BasicRAGIndex()
    index.ingest_markdown("# Passive [N3]\nrule\n## Examples\nexamples\n# Untagged\ntext", "a.md")
    assert [(chunk.section, chunk.jlpt) for chunk in index.chunks] == [("Passive", "N3"), ("Examples", "N3"),
                                                                       ("Untagged", "N5")]
