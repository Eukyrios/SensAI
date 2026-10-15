# Orchestrator: runs the agents of one learner turn. It starts with the session's agent and, while an
# agent's validated output asks for a handoff, runs the agent it names with that output as payload.
# The guards check every handoff and give every run a deadline; a stop ends the turn not ok.

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing

from ..core.agent_spec import spec_for
from ..core.events import AgentRunFinished, Error, Event, HandoffEvent
from ..core.handoff import Handoff
from .context import build_context
from .executor import AgentRun
from .guards import GuardStop, TurnGuards
from .session import Session


class Orchestrator:

    def __init__(self, session: Session, turn_id: str):
        self.session = session
        self.turn_id = turn_id
        self.guards = TurnGuards(session.config["orchestration"])
        self.runs: list[AgentRun] = []
        self.handoffs: list[Handoff] = []
        self.stopped = False

    @property
    def ok(self) -> bool:
        return bool(self.runs) and self.runs[-1].ok and not self.stopped

    @property
    def answer_text(self) -> str:
        return self.runs[-1].text if self.runs else ""

    async def events(self) -> AsyncIterator[Event]:
        handoff = None
        run = AgentRun(self.session.client, self.session.spec, self.turn_id)
        while True:
            async with aclosing(self._guarded_run(run, handoff)) as run_events:
                async for event in run_events:
                    yield event
            handoff = self._handoff_after(run)
            if handoff is None:
                return
            yield HandoffEvent(turn_id=self.turn_id, handoff=handoff)
            stop = self.guards.stop_before(handoff, [ran.spec.name for ran in self.runs])
            if stop is not None:
                yield self._stop(stop)
                return
            run = self._receiving_run(handoff)

    async def _guarded_run(self, run: AgentRun, handoff: Handoff | None) -> AsyncIterator[Event]:
        self.runs.append(run)
        deadline, stop = self.guards.run_deadline(run.spec)
        async with aclosing(run.events(build_context(self.session, run.spec, handoff))) as run_events:
            try:
                async for event in self._until(run_events, deadline):
                    yield event
            except asyncio.TimeoutError:
                yield self._stop(stop)
                yield AgentRunFinished(turn_id=self.turn_id, run_id=run.run_id, agent=run.spec.name, ok=False)

    async def _until(self, events: AsyncIterator[Event], deadline: float) -> AsyncIterator[Event]:
        while True:
            try:
                event = await asyncio.wait_for(anext(events), self.guards.seconds_left(deadline))
            except StopAsyncIteration:
                return
            yield event

    def _stop(self, stop: GuardStop) -> Error:
        self.stopped = True
        return Error(turn_id=self.turn_id, kind=stop.kind, message=stop.message)

    def _handoff_after(self, run: AgentRun) -> Handoff | None:
        request = run.data.get("handoff") if run.ok and run.data else None
        if not request:
            return None
        payload = {key: value for key, value in run.data.items() if key != "handoff"}
        handoff = Handoff(from_agent=run.spec.name, to_agent=request["to"], reason=request["reason"],
                          payload=payload, hop=len(self.handoffs) + 1, run_id=run.run_id,
                          parent_run_id=run.parent_run_id)
        self.handoffs.append(handoff)
        return handoff

    def _receiving_run(self, handoff: Handoff) -> AgentRun:
        spec = spec_for(handoff.to_agent, self.session.config)
        return AgentRun(self.session.client, spec, self.turn_id, parent_run_id=handoff.run_id)
