# Sliding-window compression (M2). When the unsummarised history takes too much of the agent's
# budget, the oldest turns are folded into a running summary by the model; the summary reaches the
# prompt through SummaryProvider and the folded turns leave the history window.

from dataclasses import dataclass

from ..core.agent_spec import AgentSpec
from ..core.messages import Message
from ..llm.ollama import OllamaError
from .session import Session

SUMMARY_PRIORITY = 0
SUMMARY_MAX_SHARE = 0.25
MIN_SUMMARY_TOKENS = 64
SUMMARY_TEMPERATURE = 0.1
SUMMARY_INSTRUCTIONS = (
    "You write notes about a language-learning conversation between a learner and a tutor. You never "
    "answer the learner and never continue the conversation: you only describe what happened in it. "
    "Keep what the learner studied, mistakes and corrections, vocabulary and grammar points, and "
    "their stated goals. Write short plain sentences in the third person, no preamble, at most {words} words."
)
SUMMARY_REQUEST = ("Earlier notes:\n{previous}\n\n<transcript>\n{transcript}\n</transcript>\n\n"
                   "Write the updated notes now: the earlier notes plus what the transcript adds.")


@dataclass(frozen=True)
class Compression:
    messages: int
    tokens_before: int
    tokens_after: int


@dataclass(frozen=True)
class SummaryProvider:
    text: str
    name: str = "summary"
    priority: int = SUMMARY_PRIORITY
    max_share: float = SUMMARY_MAX_SHARE

    def render(self, budget: int) -> str | None:
        if not self.text:
            return None
        return f"Summary of the earlier conversation: {self.text}"


def summary_budget(folded_tokens: int, window_tokens: int) -> int:
    """Half of what is folded, so a compression always frees room, within the summary's share."""
    return max(MIN_SUMMARY_TOKENS, min(folded_tokens // 2, int(window_tokens * SUMMARY_MAX_SHARE)))


def summarize(session: Session, spec: AgentSpec, older: list[Message], previous: str,
              max_tokens: int) -> str:
    transcript = "\n".join(f"{message['role']}: {message['content']}" for message in older)
    request = SUMMARY_REQUEST.format(previous=previous or "(none)", transcript=transcript)
    messages: list[Message] = [
        {"role": "system", "content": SUMMARY_INSTRUCTIONS.format(words=int(max_tokens * 0.75))},
        {"role": "user", "content": request},
    ]
    pieces = session.client.chat_stream(spec.model, messages, num_ctx=spec.num_ctx,
                                        temperature=SUMMARY_TEMPERATURE, num_predict=max_tokens)
    return "".join(piece for piece in pieces if isinstance(piece, str)).strip()


def compress(session: Session, spec: AgentSpec, window_tokens: int, estimate) -> Compression | None:
    """Fold the oldest turns into session.summary once the window passes the trigger ratio."""
    settings = session.config["memory"]["summary"]
    if not settings["enabled"]:
        return None
    pending = session.history[session.summarized:]
    keep = settings["keep_recent"]
    if len(pending) <= keep or estimate(pending) <= settings["trigger"] * window_tokens:
        return None
    older = pending[:len(pending) - keep]
    before = estimate(older) + estimate([{"role": "system", "content": session.summary}])
    try:
        text = summarize(session, spec, older, session.summary, summary_budget(before, window_tokens))
    except OllamaError:
        return None
    after = estimate([{"role": "system", "content": text}])
    if not text or after >= before:
        return None
    session.summary = text
    session.summarized += len(older)
    return Compression(len(older), before, after)
