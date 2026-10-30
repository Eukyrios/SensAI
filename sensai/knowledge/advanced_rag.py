from collections.abc import Callable
from dataclasses import asdict, dataclass, field
import re
from typing import Any

from sensai.knowledge.rag import BasicRAGIndex, DocumentChunk, _consume_llm

MINOR_WORDS = {"do", "i", "a", "an", "the", "to", "in"}
EXPLANATORY_MARKERS = ("mark", "used", "attach", "express", "rule", "means", "subject", "topic", "form")


@dataclass
class LearnerProfile:
    level: str = "N5"
    native_language: str = "en"
    goals: str = "general"


@dataclass
class ScoredChunkTrace:
    id: str
    source: str
    section: str
    jlpt: str
    initial_score: float
    rerank_score: float
    dropped: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdvancedRAGAnswer:
    question: str
    answer: str
    profile_level: str | None
    citations: list[str] = field(default_factory=list)
    retrieved_chunks: list[DocumentChunk] = field(default_factory=list)
    dropped_chunks: list[ScoredChunkTrace] = field(default_factory=list)
    rerank_trace: list[ScoredChunkTrace] = field(default_factory=list)

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "profile_level": self.profile_level,
            "citations": self.citations,
            "retrieved_chunks": [c.to_trace_dict() for c in self.retrieved_chunks],
            "dropped_chunks": [d.to_dict() for d in self.dropped_chunks],
            "rerank_scores": [r.to_dict() for r in self.rerank_trace],
        }


def resolve_profile_level(profile: Any, explicit_level: str | None = None) -> str | None:
    if explicit_level:
        return explicit_level.upper()
    if profile is None:
        return None
    if isinstance(profile, dict):
        val = profile.get("level") or profile.get("jlpt_level") or profile.get("jlpt")
        return str(val).upper() if val else None
    for attr in ("level", "jlpt_level", "jlpt"):
        val = getattr(profile, attr, None)
        if val:
            return str(val).upper()
    return None


def _all_terms(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+|[\u3040-\u30ff\u4e00-\u9fff]+", text.lower())
    return [w for w in words if w not in MINOR_WORDS]


def _bigram_set(terms: list[str]) -> set[tuple[str, str]]:
    if len(terms) < 2:
        return set()
    return {(terms[i], terms[i + 1]) for i in range(len(terms) - 1)}


def default_rerank_score(query: str, chunk: DocumentChunk) -> float:
    q_terms = _all_terms(query)
    if not q_terms:
        return chunk.score

    title_terms = _all_terms(chunk.section)
    body_terms = _all_terms(chunk.content)
    title_overlap = len(set(q_terms) & set(title_terms)) / len(set(q_terms))

    q_bigrams = _bigram_set(q_terms)
    doc_bigrams = _bigram_set(title_terms + body_terms)
    bigram_overlap = (len(q_bigrams & doc_bigrams) / len(q_bigrams)) if q_bigrams else 0.0

    lowered_body = chunk.content.lower()
    explanation_bonus = 0.15 if any(m in lowered_body for m in EXPLANATORY_MARKERS) else 0.0

    combined = (0.35 * chunk.score) + (0.35 * title_overlap) + (0.15 * bigram_overlap) + explanation_bonus
    return round(min(1.0, combined), 4)


def rerank_and_filter_chunks(
    query: str,
    candidates: list[DocumentChunk],
    relevance_threshold: float = 0.55,
    top_k: int = 2,
    reranker_fn: Callable[[str, DocumentChunk], float] | None = None,
) -> tuple[list[DocumentChunk], list[ScoredChunkTrace], list[ScoredChunkTrace]]:
    scorer = reranker_fn or default_rerank_score
    evaluated = [(c, c.score, round(float(scorer(query, c)), 4)) for c in candidates]
    evaluated.sort(key=lambda item: item[2], reverse=True)

    kept: list[DocumentChunk] = []
    dropped: list[ScoredChunkTrace] = []
    full_trace: list[ScoredChunkTrace] = []

    for chunk, init_score, r_score in evaluated:
        is_dropped = r_score < relevance_threshold or len(kept) >= top_k
        trace_entry = ScoredChunkTrace(
            id=chunk.id,
            source=chunk.source,
            section=chunk.section,
            jlpt=chunk.jlpt,
            initial_score=init_score,
            rerank_score=r_score,
            dropped=is_dropped,
        )
        full_trace.append(trace_entry)
        if r_score < relevance_threshold:
            dropped.append(trace_entry)
        elif len(kept) < top_k:
            kept.append(
                DocumentChunk(
                    id=chunk.id,
                    content=chunk.content,
                    source=chunk.source,
                    section=chunk.section,
                    jlpt=chunk.jlpt,
                    score=r_score,
                )
            )

    return kept, dropped, full_trace


def retrieve_advanced(
    question: str,
    index: BasicRAGIndex,
    profile: Any = None,
    max_jlpt: str | None = None,
    relevance_threshold: float = 0.55,
    top_k: int = 2,
    candidate_k: int = 6,
    reranker_fn: Callable[[str, DocumentChunk], float] | None = None,
) -> tuple[list[DocumentChunk], list[ScoredChunkTrace], list[ScoredChunkTrace], str | None]:
    level = resolve_profile_level(profile, explicit_level=max_jlpt)
    candidates = index.retrieve(question, max_jlpt=level, top_k=candidate_k, min_score=0.15)
    kept, dropped, full_trace = rerank_and_filter_chunks(
        question,
        candidates,
        relevance_threshold=relevance_threshold,
        top_k=top_k,
        reranker_fn=reranker_fn,
    )
    return kept, dropped, full_trace, level


def _format_rag_reply(question: str, kept: list[DocumentChunk], citations: list[str], client: Any, model: str) -> str:
    if client is None:
        return f"{kept[0].content} {citations[0]}"
    context_block = "\n".join(f"{c.citation()}: {c.content}" for c in kept)
    prompt = f"Use this context:\n{context_block}\nQuestion: {question}\nInclude citation {citations[0]}."
    reply = _consume_llm(client, model, prompt).strip()
    if citations[0] not in reply:
        return f"{reply} {citations[0]}".strip()
    return reply


def answer_with_advanced_rag(
    question: str,
    index: BasicRAGIndex,
    profile: Any = None,
    client: Any = None,
    model: str = "qwen3.5:9b",
    max_jlpt: str | None = None,
    relevance_threshold: float = 0.55,
    top_k: int = 2,
    reranker_fn: Callable[[str, DocumentChunk], float] | None = None,
) -> AdvancedRAGAnswer:
    kept, dropped, full_trace, level = retrieve_advanced(
        question=question,
        index=index,
        profile=profile,
        max_jlpt=max_jlpt,
        relevance_threshold=relevance_threshold,
        top_k=top_k,
        reranker_fn=reranker_fn,
    )
    if not kept:
        fallback = "This topic is not covered at your current JLPT level in the grammar corpus."
        return AdvancedRAGAnswer(
            question=question,
            answer=fallback,
            profile_level=level,
            citations=[],
            retrieved_chunks=[],
            dropped_chunks=dropped,
            rerank_trace=full_trace,
        )

    citations = [chunk.citation() for chunk in kept]
    reply = _format_rag_reply(question, kept, citations, client, model)
    return AdvancedRAGAnswer(
        question=question,
        answer=reply,
        profile_level=level,
        citations=citations,
        retrieved_chunks=kept,
        dropped_chunks=dropped,
        rerank_trace=full_trace,
    )
