from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any

JLPT_ORDER = {"N5": 1, "N4": 2, "N3": 3, "N2": 4, "N1": 5}
STOPWORDS = {"when", "do", "i", "use", "what", "is", "the", "difference", "between", "how", "to", "in", "a", "an", "vs"}


@dataclass
class DocumentChunk:
    id: str
    content: str
    source: str
    section: str
    jlpt: str = "N5"
    score: float = 0.0

    def citation(self) -> str:
        return f"[{self.source} § {self.section}]"

    def to_trace_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: list[str] = field(default_factory=list)
    retrieved_chunks: list[DocumentChunk] = field(default_factory=list)

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "retrieved_chunks": [c.to_trace_dict() for c in self.retrieved_chunks],
        }


def _tokenize(text: str) -> set[str]:
    ascii_words = {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in STOPWORDS and len(w) > 1
    }
    jp_chars = set(re.findall(r"[\u3040-\u30ff\u4e00-\u9fff]", text))
    return ascii_words | jp_chars


def _allowed_jlpt(chunk_level: str, max_level: str | None) -> bool:
    if not max_level:
        return True
    target = JLPT_ORDER.get(max_level.upper(), 5)
    current = JLPT_ORDER.get(chunk_level.upper(), 5)
    return current <= target


def _parse_header(line: str, default_jlpt: str) -> tuple[str, str]:
    clean = line.lstrip("#").strip()
    match = re.search(r"\[(N[1-5])\]", clean, re.IGNORECASE)
    if match:
        level = match.group(1).upper()
        section = re.sub(r"\s*\[N[1-5]\]\s*", "", clean, flags=re.IGNORECASE).strip()
        return section, level
    return clean, default_jlpt


def _score_chunk(query_tokens: set[str], chunk: DocumentChunk) -> float:
    if not query_tokens:
        return 0.0
    chunk_tokens = _tokenize(f"{chunk.section} {chunk.content}")
    overlap = query_tokens & chunk_tokens
    if not overlap:
        return 0.0
    return round(len(overlap) / len(query_tokens), 4)


class BasicRAGIndex:
    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []

    @property
    def chunks(self) -> list[DocumentChunk]:
        return list(self._chunks)

    def add_chunk(self, content: str, source: str, section: str, jlpt: str = "N5") -> DocumentChunk:
        chunk_id = f"{source}:{len(self._chunks) + 1}"
        chunk = DocumentChunk(
            id=chunk_id,
            content=content.strip(),
            source=source,
            section=section,
            jlpt=jlpt.upper(),
        )
        self._chunks.append(chunk)
        return chunk

    def ingest_markdown(self, text: str, source: str, default_jlpt: str = "N5") -> list[DocumentChunk]:
        added: list[DocumentChunk] = []
        current_section = "Introduction"
        current_jlpt = default_jlpt.upper()
        buffer: list[str] = []

        def flush() -> None:
            body = "\n".join(buffer).strip()
            if body:
                added.append(self.add_chunk(body, source, current_section, current_jlpt))
            buffer.clear()

        for line in text.splitlines():
            if line.strip().startswith("#"):
                flush()
                current_section, current_jlpt = _parse_header(line, default_jlpt)
            else:
                buffer.append(line)
        flush()
        return added

    def ingest_file(self, path: str | Path, default_jlpt: str = "N5") -> list[DocumentChunk]:
        file_path = Path(path)
        content = file_path.read_text(encoding="utf-8")
        return self.ingest_markdown(content, source=file_path.name, default_jlpt=default_jlpt)

    def ingest_directory(self, directory: str | Path, default_jlpt: str = "N5") -> list[DocumentChunk]:
        dir_path = Path(directory)
        ingested: list[DocumentChunk] = []
        for file_path in sorted(dir_path.glob("*.md")):
            ingested.extend(self.ingest_file(file_path, default_jlpt=default_jlpt))
        return ingested

    def retrieve(self, query: str, max_jlpt: str | None = None, top_k: int = 3, min_score: float = 0.2) -> list[DocumentChunk]:
        query_tokens = _tokenize(query)
        scored: list[DocumentChunk] = []

        for chunk in self._chunks:
            if not _allowed_jlpt(chunk.jlpt, max_jlpt):
                continue
            score = _score_chunk(query_tokens, chunk)
            if score >= min_score:
                scored.append(
                    DocumentChunk(
                        id=chunk.id,
                        content=chunk.content,
                        source=chunk.source,
                        section=chunk.section,
                        jlpt=chunk.jlpt,
                        score=score,
                    )
                )

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]


def _consume_llm(client: Any, model: str, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(client, "chat_stream"):
        return "".join(c for c in client.chat_stream(model, messages) if isinstance(c, str))
    if hasattr(client, "chat"):
        return client.chat(model, messages)
    return ""


def answer_with_rag(
    question: str,
    index: BasicRAGIndex,
    client: Any = None,
    model: str = "qwen3.5:9b",
    max_jlpt: str | None = None,
    top_k: int = 2,
) -> RAGAnswer:
    retrieved = index.retrieve(question, max_jlpt=max_jlpt, top_k=top_k)
    if not retrieved:
        fallback = "This topic is not covered in the grammar corpus, so here is a general explanation without a source citation."
        if client is not None:
            llm_reply = _consume_llm(client, model, f"Answer without citing any corpus source: {question}")
            if llm_reply:
                fallback = llm_reply
        return RAGAnswer(question=question, answer=fallback, citations=[], retrieved_chunks=[])

    citations = [chunk.citation() for chunk in retrieved]
    context_block = "\n".join(f"{c.citation()}: {c.content}" for c in retrieved)

    if client is not None:
        prompt = f"Use this context:\n{context_block}\nQuestion: {question}\nInclude citation {citations[0]}."
        reply = _consume_llm(client, model, prompt).strip()
        if citations[0] not in reply:
            reply = f"{reply} {citations[0]}".strip()
    else:
        reply = f"{retrieved[0].content} {citations[0]}"

    return RAGAnswer(
        question=question,
        answer=reply,
        citations=citations,
        retrieved_chunks=retrieved,
    )