# run_turn(): one learner turn as a stream of events. It saves the question, lets the orchestrator
# run the agents, then keeps the last agent's answer, with the turn and its agent runs, or rolls the
# question back. Once the answer and TurnFinished are out, the judge's verdict follows in the same
# stream, while the curriculum agent updates the learner's srs_cards in the background
# (Session.settle() waits for it).

import asyncio
import re
import time
from collections.abc import AsyncIterator
from contextlib import aclosing

from ..core.events import (Error, Event, FaithfulnessChecked, GuardrailTriggered, Token, TurnFinished,
                           TurnStarted)
from ..core.ids import new_id
from ..llm.ollama import ChatStats
from ..trust import guardrails
from ..trust.eval import LLMJudge, SourceCollector, faithfulness_event
from .curriculum import run_curriculum
from .executor import AgentRun, sum_stats
from .orchestrator import Orchestrator
from .session import Session


async def run_turn(session: Session, text: str) -> AsyncIterator[Event]:
    text = printable(text)
    turn = _Turn(session)
    yield TurnStarted(turn_id=turn.turn_id, session_id=session.id, user_input=text)
    learner_text = text.strip()
    if not learner_text:
        yield Error(turn_id=turn.turn_id, kind="empty_input", message="Empty message")
        yield turn.finished_event()
        return
    verdict = guardrails.check_input(learner_text, session.config)
    if verdict.action != "allow":
        yield turn.guardrail_event(verdict)
    if verdict.action == "block":
        yield Token(turn_id=turn.turn_id, run_id=turn.turn_id, text=guardrails.refusal_for(verdict.rule_triggered))
        yield turn.finished_event()
        return
    clean_text = verdict.filtered_text
    turn.save_question(session.router.strip_override(clean_text) if session.router else clean_text)
    async with aclosing(turn.answer(clean_text)) as events:
        async for event in redact_tokens(events, turn):
            turn.sources.observe(event)
            yield event
    yield turn.finished_event()
    if turn.crew.ok:
        session.run_in_background(run_curriculum(session, turn.turn_id))
        verdict_event = await turn.check_faithfulness(learner_text)
        if verdict_event is not None:
            yield verdict_event


# What may still grow into PII is kept back: the last word (an email) and the digit groups before it
# (a phone number or a card). Everything before that is released.
WORD_LIMIT_CHARS = 40
RISKY_TAIL = re.compile(r"(?:[\d+()\-.]+\s+)*\S{0,40}$")


async def redact_tokens(events: AsyncIterator[Event], turn: "_Turn") -> AsyncIterator[Event]:
    """Masks PII in the reply before it reaches a surface. Other events pass through in order, and tokens
    are passed on untouched unless something was masked in them."""
    config = turn.session.config
    rules = guardrails.rules_from(config)
    if not rules["enabled"] or not rules["redact_pii_output"]:
        async for event in events:
            yield event
        return
    pending: list[Token] = []
    redacted = False

    def release(count: int) -> list[Token]:
        nonlocal redacted
        group, rest = pending[:count], pending[count:]
        pending[:] = rest
        text = "".join(token.text for token in group)
        verdict = guardrails.check_output(text, config)
        if verdict.action != "redact":
            return group
        redacted = True
        return [Token(turn_id=group[0].turn_id, run_id=group[0].run_id, text=verdict.filtered_text, ts=group[0].ts)]

    async for event in events:
        if not isinstance(event, Token):
            if pending:
                for token in release(len(pending)):
                    yield token
            yield event
            continue
        if pending and pending[-1].run_id != event.run_id:
            for token in release(len(pending)):
                yield token
        pending.append(event)
        count = _tokens_to_release(pending)
        if count:
            for token in release(count):
                yield token
    if pending:
        for token in release(len(pending)):
            yield token
    if redacted:
        yield turn.output_guardrail_event()


def _tokens_to_release(pending: list[Token]) -> int:
    """How many of the oldest pending tokens end before the risky tail of the text."""
    text = "".join(token.text for token in pending)
    cut = RISKY_TAIL.search(text).start()
    ends, length = [], 0
    for token in pending:
        length += len(token.text)
        if length > cut:
            break
        ends.append(length)
    while ends and _inside_a_word(text, ends[-1]):
        ends.pop()
    return len(ends)


def _inside_a_word(text: str, position: int) -> bool:
    """True when cutting at `position` would split a word short enough to be an email or a key."""
    if position >= len(text) or text[position - 1].isspace() or text[position].isspace():
        return False
    left = max(text.rfind(" ", 0, position), text.rfind("\\n", 0, position)) + 1
    spaces = [i for i in (text.find(" ", position), text.find("\\n", position)) if i != -1]
    right = min(spaces) if spaces else len(text)
    return right - left <= WORD_LIMIT_CHARS


class _Turn:

    def __init__(self, session: Session):
        self.session = session
        self.turn_id = new_id()
        self.started_ns = time.perf_counter_ns()
        self.crew = Orchestrator(session, self.turn_id)
        self.sources = SourceCollector()
        self.question_id: str | None = None
        self.question_parent_id: str | None = None

    async def answer(self, question: str) -> AsyncIterator[Event]:
        try:
            async with aclosing(self.crew.events(question)) as crew_events:
                async for event in crew_events:
                    yield event
        except (asyncio.CancelledError, KeyboardInterrupt):
            self.keep_partial_answer(self.crew.answer_text)
            raise
        except BaseException:
            self.drop_question()
            raise
        if self.crew.ok:
            self.record_runs(self.save_answer(self.crew.answer_text))
            self.crew.record_errors()
        else:
            self.drop_question()

    async def check_faithfulness(self, question: str) -> FaithfulnessChecked | None:
        """EV1: the judge checks the reply against the passages and tool results it was built on. It runs
        once the reply is out, in a thread, and never fails the turn."""
        settings = self.session.config["eval"]
        run = self.crew.runs[-1]
        reply = self.session.history[-1]["content"]
        if not settings["enabled"] or not self.sources or not reply.strip() or run.spec.output_schema is not None:
            return None
        model = settings["judge_model"] or run.spec.model
        judge = LLMJudge(self.session.client, model, settings["max_context_chars"])
        try:
            report = await asyncio.to_thread(judge.verify_response, reply, self.sources.context, question)
        except Exception as error:  # the judge is optional: whatever goes wrong is reported, not raised
            return FaithfulnessChecked(turn_id=self.turn_id, faithfulness=None, judge_model=model, error=str(error))
        return faithfulness_event(self.turn_id, report, model)

    def save_question(self, text: str) -> None:
        self.session.history.append({"role": "user", "content": text})
        self.question_parent_id = self.session.leaf
        self.question_id = self.session.record("user", text)

    def drop_question(self) -> None:
        self.session.history.pop()
        self.session.discard(self.question_id, self.question_parent_id)

    def save_answer(self, text: str) -> str:
        text = guardrails.check_output(text, self.session.config).filtered_text
        self.session.history.append({"role": "assistant", "content": text})
        return self.session.record("assistant", text)

    def record_runs(self, answer_id: str) -> None:
        stats = self._total_stats()
        self.session.repo.record_turn(
            self.session.id, self.question_id, answer_id, turn_id=self.turn_id,
            latency_ms=(time.perf_counter_ns() - self.started_ns) // 1_000_000, tokens=_tokens(stats),
        )
        for run in self.crew.runs:
            self._record_run(run)

    def _record_run(self, run: AgentRun) -> None:
        self.session.repo.record_agent_run(
            self.turn_id, "agent", _run_status(run), output=run.text, duration_ms=_duration_ms(run.stats),
            parent_run_id=run.parent_run_id, run_id=run.run_id, agent=run.spec.name,
            prompt_hash=run.spec.prompt_hash, tokens=_tokens(run.stats),
        )

    def keep_partial_answer(self, text: str) -> None:
        if text:
            self.save_answer(text)
        else:
            self.drop_question()

    def guardrail_event(self, verdict: guardrails.GuardrailResult) -> GuardrailTriggered:
        return GuardrailTriggered(turn_id=self.turn_id, side="input", rule=verdict.rule_triggered or "guardrail",
                                  action=verdict.action)

    def output_guardrail_event(self) -> GuardrailTriggered:
        return GuardrailTriggered(turn_id=self.turn_id, side="output", rule="pii_leakage", action="redact")

    def finished_event(self) -> TurnFinished:
        stats = self._total_stats() or ChatStats()
        model = self.crew.runs[-1].spec.model if self.crew.runs else self.session.spec.model
        return TurnFinished(turn_id=self.turn_id, model=model, ttft_ms=self._first_output_ms(),
                            prompt_eval_count=stats.prompt_eval_count, eval_count=stats.eval_count,
                            eval_duration_ns=stats.eval_duration_ns,
                            total_duration_ns=time.perf_counter_ns() - self.started_ns)

    def _total_stats(self) -> ChatStats | None:
        stats = None
        for run in self.crew.runs:
            stats = sum_stats(stats, run.stats)
        return stats

    def _first_output_ms(self) -> float | None:
        outputs = [run.first_output_ns for run in self.crew.runs if run.first_output_ns is not None]
        return (min(outputs) - self.started_ns) / 1e6 if outputs else None


def _run_status(run: AgentRun) -> str:
    if run.ok:
        return "ok"
    return run.error_kind or "error"


def _duration_ms(stats: ChatStats | None) -> int | None:
    return stats.total_duration_ns // 1_000_000 if stats else None


def _tokens(stats: ChatStats | None) -> int | None:
    return stats.prompt_eval_count + stats.eval_count if stats else None


def printable(text: str) -> str:
    """The message as valid UTF-8 without control characters: bytes that are not UTF-8 (read as surrogates
    from a pipe) become U+FFFD and NUL or escape characters are dropped, so no log or request breaks on them."""
    try:
        text = text.encode("utf-8", "surrogateescape").decode("utf-8", "replace")
    except UnicodeEncodeError:
        text = text.encode("utf-8", "replace").decode("utf-8")
    return "".join(char for char in text if char in "\n\t" or ord(char) >= 32 and ord(char) != 127)
