from sensai.knowledge.advanced_rag import LearnerProfile, answer_with_advanced_rag
from sensai.knowledge.rag import BasicRAGIndex, answer_with_rag


def _build_competing_sources_index() -> BasicRAGIndex:
    index = BasicRAGIndex()
    index.ingest_markdown(
        "# Casual Chat Log [N5]\nStudents often ask about は and が in class discussions.",
        source="chat_notes.md",
    )
    index.ingest_markdown(
        "# Particles は vs が [N5]\nUse は (wa) to mark the topic and が (ga) to mark the grammatical subject.",
        source="grammar_reference.md",
    )
    return index


def test_r2_1_reranking_and_threshold_dropping_in_trace():
    index = _build_competing_sources_index()
    result = answer_with_advanced_rag("when do I use は vs が?", index, relevance_threshold=0.55, top_k=2)
    trace = result.to_trace_dict()

    assert len(result.retrieved_chunks) == 1
    assert result.retrieved_chunks[0].source == "grammar_reference.md"
    assert len(trace["dropped_chunks"]) == 1
    assert trace["dropped_chunks"][0]["source"] == "chat_notes.md"
    assert trace["dropped_chunks"][0]["rerank_score"] < 0.55
    assert trace["dropped_chunks"][0]["dropped"] is True


def test_r2_1_reranked_answer_cites_better_source_than_r1():
    index = _build_competing_sources_index()
    question = "when do I use は vs が?"

    r1_result = answer_with_rag(question, index, top_k=1)
    r2_result = answer_with_advanced_rag(question, index, top_k=1, relevance_threshold=0.55)

    assert r1_result.citations[0] == "[chat_notes.md § Casual Chat Log]"
    assert r2_result.citations[0] == "[grammar_reference.md § Particles は vs が]"
    assert r2_result.citations[0] != r1_result.citations[0]


def test_r2_2_chunks_filtered_by_learner_profile_level_excludes_n1():
    index = BasicRAGIndex()
    index.ingest_markdown(
        "# Basic Negation ない [N5]\nUse ない (nai) to form plain negative verbs.\n"
        "# Formal Compulsion ざるを得ない [N1]\nUse ざるを得ない with ない stem to express 'cannot help doing'.",
        source="negation.md",
    )

    n5_profile = LearnerProfile(level="N5")
    result = answer_with_advanced_rag("negative form ない", index, profile=n5_profile, relevance_threshold=0.4)

    assert len(result.retrieved_chunks) == 1
    assert all(c.jlpt == "N5" for c in result.retrieved_chunks)
    assert "ざるを得ない" not in result.answer


def test_r2_2_changing_profile_level_changes_retrievable_chunks():
    index = BasicRAGIndex()
    index.ingest_markdown(
        "# Basic Request てください [N5]\nAttach てください (te kudasai) to a verb to make a polite request.\n"
        "# Conditional Request たらどうですか [N4]\nUse たらどうですか to make a polite request or suggestion.\n"
        "# Honorific Request お願い申し上げます [N1]\nUsed in formal letters to make a polite request.",
        source="requests.md",
    )

    profile = LearnerProfile(level="N5")
    res_n5 = answer_with_advanced_rag("polite request", index, profile=profile, relevance_threshold=0.3, top_k=5)
    assert {c.jlpt for c in res_n5.retrieved_chunks} == {"N5"}

    profile.level = "N4"
    res_n4 = answer_with_advanced_rag("polite request", index, profile=profile, relevance_threshold=0.3, top_k=5)
    assert {c.jlpt for c in res_n4.retrieved_chunks} == {"N5", "N4"}

    profile.level = "N1"
    res_n1 = answer_with_advanced_rag("polite request", index, profile=profile, relevance_threshold=0.3, top_k=5)
    assert {c.jlpt for c in res_n1.retrieved_chunks} == {"N5", "N4", "N1"}