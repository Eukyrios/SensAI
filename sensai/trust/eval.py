"""Automated Evaluation & Hallucination Detection (EV1).

Claim-by-claim verification of a reply against the sources the agent had (corpus passages and tool
results), plus a relevance grade for the question. The judge talks to Ollama through `chat_stream`,
like every agent, and a failure is reported as an error, never as a score.
"""
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

from ..core.events import Event, FaithfulnessChecked, Retrieved, ToolResult
from ..core.messages import Message
from ..llm.ollama import OllamaClient, OllamaError, Thought, ToolRequest

ClaimStatus = Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIED"]
STATUSES = ("SUPPORTED", "CONTRADICTED", "UNVERIFIED")
JUDGE_NUM_CTX = 8192
MAX_RELEVANCE = 10

SYSTEM_PROMPT = (
    "You are an impartial judge of factual consistency. You never answer the question yourself: you only "
    "check a reply against the sources it was given. Reply with one JSON object and nothing else."
)


@dataclass
class ClaimVerification:
    claim: str
    status: ClaimStatus
    reasoning: str


@dataclass
class VerificationReport:
    claims: list[ClaimVerification] = field(default_factory=list)
    relevance: float | None = None
    error: str | None = None

    @property
    def total_claims(self) -> int:
        return len(self.claims)

    @property
    def supported_claims(self) -> int:
        return sum(1 for c in self.claims if c.status == "SUPPORTED")

    @property
    def contradicted_claims(self) -> int:
        return sum(1 for c in self.claims if c.status == "CONTRADICTED")

    @property
    def unverified_claims(self) -> int:
        return sum(1 for c in self.claims if c.status == "UNVERIFIED")

    @property
    def faithfulness_score(self) -> float:
        if self.total_claims == 0:
            return 1.0
        return round(self.supported_claims / self.total_claims, 2)

    @property
    def flagged(self) -> list[ClaimVerification]:
        return [claim for claim in self.claims if claim.status != "SUPPORTED"]


class LLMJudge:
    """Evaluates agent responses for factual consistency and hallucination detection."""

    def __init__(self, client: OllamaClient, judge_model: str | None = None, max_context_chars: int = 6000) -> None:
        self.client = client
        self.judge_model = judge_model
        self.max_context_chars = max_context_chars

    def verify_response(self, response_text: str, context: str, question: str = "") -> VerificationReport:
        """Verifies a response against reference context claim-by-claim, and grades its relevance."""
        if not response_text.strip() or not context.strip():
            return VerificationReport()
        if not self.judge_model:
            return VerificationReport(error="no judge model set")
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": self._prompt(response_text, context[:self.max_context_chars], question)},
        ]
        try:
            return _parse_report(self._ask(messages))
        except (OllamaError, ValueError) as error:
            return VerificationReport(error=str(error))

    def _ask(self, messages: list[Message]) -> str:
        text = ""
        for piece in self.client.chat_stream(self.judge_model, messages, num_ctx=JUDGE_NUM_CTX, temperature=0.0,
                                             think=False, format="json"):
            if not isinstance(piece, (Thought, ToolRequest)):
                text += piece
        return text

    @staticmethod
    def _prompt(response_text: str, context: str, question: str) -> str:
        return f"""Check the REPLY against the SOURCES.

Step 1: split the REPLY into standalone factual claims (skip greetings and advice).
Step 2: give each claim one status:
- "SUPPORTED": stated or directly entailed by the SOURCES.
- "CONTRADICTED": the SOURCES say the opposite.
- "UNVERIFIED": the SOURCES neither confirm nor deny it (facts from memory count as UNVERIFIED).
Step 3: grade from 0 to {MAX_RELEVANCE} how well the REPLY answers the QUESTION.

SOURCES:
{context}

QUESTION:
{question or "(not given)"}

REPLY:
{response_text}

Answer with this JSON shape only:
{{"claims": [{{"claim": "...", "status": "SUPPORTED", "reasoning": "short"}}], "relevance": 0}}"""


def _parse_report(raw: str) -> VerificationReport:
    parsed = _load_json(raw)
    claims = [
        ClaimVerification(claim=str(item.get("claim", "")), status=_status(item.get("status")),
                          reasoning=str(item.get("reasoning", "")))
        for item in parsed.get("claims", []) if isinstance(item, dict)
    ]
    return VerificationReport(claims=claims, relevance=_relevance(parsed.get("relevance")))


def _load_json(raw: str) -> dict:
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError(f"the judge did not answer with JSON: {text[:80]!r}")
    parsed = json.loads(text[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("the judge's JSON is not an object")
    return parsed


def _status(value: object) -> ClaimStatus:
    status = str(value).strip().upper()
    return status if status in STATUSES else "UNVERIFIED"  # type: ignore[return-value]


def _relevance(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(min(max(value, 0), MAX_RELEVANCE) / MAX_RELEVANCE, 2)


class SourceCollector:
    """What a turn's agents had to ground their reply on: the passages kept by retrieval and the
    results of the tools that succeeded."""

    def __init__(self) -> None:
        self.passages: list[str] = []

    def observe(self, event: Event) -> None:
        match event:
            case Retrieved(kept=kept):
                self.passages += [f"[{chunk['source']} § {chunk['section']}]\n{chunk['content']}"
                                  for chunk in kept if chunk.get("content")]
            case ToolResult(name=name, result=result, ok=True) if result.strip():
                self.passages.append(f"[tool {name}]\n{result}")

    @property
    def context(self) -> str:
        return "\n\n".join(self.passages)

    def __bool__(self) -> bool:
        return bool(self.passages)


def faithfulness_event(turn_id: str, report: VerificationReport, model: str) -> FaithfulnessChecked:
    return FaithfulnessChecked(
        turn_id=turn_id, faithfulness=None if report.error else report.faithfulness_score,
        relevance=report.relevance, supported=report.supported_claims, contradicted=report.contradicted_claims,
        unverified=report.unverified_claims, flagged=[asdict(claim) for claim in report.flagged],
        judge_model=model, error=report.error)
