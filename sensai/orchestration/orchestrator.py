# Orchestrator: runs the agents of one learner turn. It starts with the agent the router picks (the
# session's agent when routing is off) and, while an agent's validated output asks for a handoff,
# runs the agent it names with that output as payload. Each run's prompt is resolved by the prompt
# store first: saved by hash, or swapped for its pinned version.
# The guards check every handoff and give every run a deadline. When the chain fails, the fallback
# agent (the sensei) answers instead, unless Ollama is down, the budget is spent or it already ran.

import asyncio
from collections.abc import AsyncIterator
from contextlib import aclosing

from ..core.agent_spec import AgentSpec, spec_for
from ..core.errors import ErrorKind
from ..core.events import AgentRunFinished, Error, Event, HandoffEvent, RouteDecided
from ..core.handoff import Handoff
from .context import build_context
from .executor import AgentRun
from .guards import GuardStop, TurnGuards
from .router import RouteDecision
from .session import Session

NO_FALLBACK: tuple[ErrorKind, ...] = ("connection", "turn_budget")


class Orchestrator:

    def __init__(self, session: Session, turn_id: str):
        self.session = session
        self.turn_id = turn_id
        self.guards = TurnGuards(session.config["orchestration"])
        self.runs: list[AgentRun] = []
        self.handoffs: list[Handoff] = []
        self.stop_kind: ErrorKind | None = None
        self.route: RouteDecision | None = None

    @property
    def ok(self) -> bool:
        return bool(self.runs) and self.runs[-1].ok and self.stop_kind is None

    @property
    def answer_text(self) -> str:
        return self.runs[-1].text if self.runs else ""

    async def events(self, question: str) -> AsyncIterator[Event]:
        first_agent = await self._first_agent(question)
        if self.route is not None:
            yield RouteDecided(turn_id=self.turn_id, agent=self.route.agent, tier=self.route.tier,
                               confidence=self.route.confidence, reason=self.route.reason)
        async with aclosing(self._handoff_chain(first_agent)) as chain_events:
            async for event in chain_events:
                yield event
        fallback = self._fallback_run()
        if fallback is None:
            return
        self.stop_kind = None
        last_handoff = self.handoffs[-1] if self.handoffs else None
        async with aclosing(self._guarded_run(fallback, last_handoff)) as fallback_events:
            async for event in fallback_events:
                yield event

    async def _first_agent(self, question: str) -> AgentSpec:
        if self.session.router is None:
            return self.session.spec
        self.route = await asyncio.to_thread(self.session.router.route, question)
        if self.route.agent == self.session.spec.name:
            return self.session.spec
        return spec_for(self.route.agent, self.session.config)

    async def _handoff_chain(self, first_agent: AgentSpec) -> AsyncIterator[Event]:
        handoff = None
        run = AgentRun(self.session.client, first_agent, self.turn_id)
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
        run.spec = self.session.prompts.resolve(run.spec)
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
        self.stop_kind = stop.kind
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

    def _fallback_run(self) -> AgentRun | None:
        name = self.session.config["agents"]["fallback"]
        failed = self.runs[-1]
        if self.ok or (self.stop_kind or failed.error_kind) in NO_FALLBACK:
            return None
        if name in [ran.spec.name for ran in self.runs]:
            return None
        spec = spec_for(name, self.session.config)
        return AgentRun(self.session.client, spec, self.turn_id, parent_run_id=failed.run_id)

    def _receiving_run(self, handoff: Handoff) -> AgentRun:
        spec = spec_for(handoff.to_agent, self.session.config)
        return AgentRun(self.session.client, spec, self.turn_id, parent_run_id=handoff.run_id)
