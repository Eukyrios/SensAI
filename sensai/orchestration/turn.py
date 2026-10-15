# run_turn(): one learner turn as a stream of events. It saves the question, lets the agent executor
# answer, then keeps or rolls back the history depending on how the agent run ended.

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import aclosing

from ..core.events import Error, Event, GuardrailTriggered, TurnFinished, TurnStarted
from ..core.ids import new_id
from ..core.messages import Message
from ..trust import guardrails
from .executor import AgentRun, sum_stats, system_prompt
from .session import Session


async def run_turn(session: Session, text: str) -> AsyncIterator[Event]:
    turn = _Turn(session)
    yield TurnStarted(turn_id=turn.turn_id, session_id=session.id, user_input=text)
    learner_text = text.strip()
    if not learner_text:
        yield Error(turn_id=turn.turn_id, kind="empty_input", message="Empty message")
        yield turn.finished_event()
        return
    clean_text = guardrails.filter_input(learner_text, session.config)
    if clean_text != learner_text:
        yield turn.guardrail_redacted_event()
    turn.save_question(clean_text)
    async with aclosing(turn.run_agent(AgentRun(session.client, session.spec, turn.turn_id))) as events:
        async for event in events:
            yield event
    yield turn.finished_event()


class _Turn:

    def __init__(self, session: Session):
        self.session = session
        self.turn_id = new_id()
        self.started_ns = time.perf_counter_ns()
        self.runs: list[AgentRun] = []
        self.question_id: str | None = None
        self.question_parent_id: str | None = None

    async def run_agent(self, run: AgentRun) -> AsyncIterator[Event]:
        self.runs.append(run)
        try:
            async with aclosing(run.events(build_context(self.session))) as run_events:
                async for event in run_events:
                    yield event
        except (asyncio.CancelledError, KeyboardInterrupt):
            self.keep_partial_answer(run.text)
            raise
        except BaseException:
            self.drop_question()
            raise
        if run.ok:
            self.save_answer(run.text)
        else:
            self.drop_question()

    def save_question(self, text: str) -> None:
        self.session.history.append({"role": "user", "content": text})
        self.question_parent_id = self.session.leaf
        self.question_id = self.session.record("user", text)

    def drop_question(self) -> None:
        self.session.history.pop()
        self.session.discard(self.question_id, self.question_parent_id)

    def save_answer(self, text: str) -> None:
        self.session.history.append({"role": "assistant", "content": text})
        self.session.record("assistant", text)

    def keep_partial_answer(self, text: str) -> None:
        if text:
            self.save_answer(text)
        else:
            self.drop_question()

    def guardrail_redacted_event(self) -> GuardrailTriggered:
        return GuardrailTriggered(turn_id=self.turn_id, side="input", rule="filter_input", action="redact")

    def finished_event(self) -> TurnFinished:
        stats = None
        for run in self.runs:
            stats = sum_stats(stats, run.stats)
        model = self.runs[-1].spec.model if self.runs else self.session.spec.model
        if stats is None:
            return TurnFinished(turn_id=self.turn_id, model=model, ttft_ms=self._first_output_ms(),
                                total_duration_ns=time.perf_counter_ns() - self.started_ns)
        return TurnFinished(turn_id=self.turn_id, model=model, ttft_ms=self._first_output_ms(),
                            prompt_eval_count=stats.prompt_eval_count, eval_count=stats.eval_count,
                            total_duration_ns=stats.total_duration_ns)

    def _first_output_ms(self) -> float | None:
        outputs = [run.first_output_ns for run in self.runs if run.first_output_ns is not None]
        return (min(outputs) - self.started_ns) / 1e6 if outputs else None


def build_context(session: Session) -> list[Message]:
    max_turns = session.config["memory"]["max_history_turns"]
    profile = [session.profile] if session.profile else []
    return ([{"role": "system", "content": system_prompt(session.spec)}]
            + profile
            + session.history[-max_turns * 2:])
