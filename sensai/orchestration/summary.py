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
SUMMARY_TEMPERATURE = 0.1
SUMMARY_INSTRUCTIONS = (
    "You compress a language-learning conversation. Update the running summary with the new turns. "
    "Keep what the learner studied, mistakes and corrections, vocabulary and grammar points, and "
    "their stated goals. Write short plain sentences, no preamble, at most {words} words."
)


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


def summarize(session: Session, spec: AgentSpec, older: list[Message], previous: str,
              max_tokens: int) -> str:
    transcript = "\n".join(f"{message['role']}: {message['content']}" for message in older)
    request = f"Running summary:\n{previous or '(none)'}\n\nNew turns:\n{transcript}"
    messages: list[Message] = [
        {"role": "system", "content": SUMMARY_INSTRUCTIONS.format(words=int(max_tokens * 0.75))},
        {"role": "user", "content": request},
    ]
    pieces = session.client.chat_stream(spec.model, messages, num_ctx=spec.num_ctx,
                                        temperature=SUMMARY_TEMPERATURE)
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
    try:
        text = summarize(session, spec, older, session.summary,
                         int(window_tokens * SUMMARY_MAX_SHARE))
    except OllamaError:
        return None
    if not text:
        return None
    session.summary = text
    session.summarized += len(older)
    return Compression(len(older), estimate(older), estimate([{"role": "system", "content": text}]))
