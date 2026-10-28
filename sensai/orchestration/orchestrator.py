# Orchestrator: runs the agents of one learner turn. It starts with the agent the router picks (the
# session's agent when routing is off) and, while an agent's validated output asks for a handoff,
# runs the agent it names with that output as payload. Each run gets the session's persona, then its
# prompt is resolved by the prompt store: saved by hash, or swapped for its pinned version.
# The guards check every handoff and give every run a deadline. When the chain fails, the fallback
# agent (the sensei) answers instead, unless Ollama is down, the budget is spent or it already ran.
# A run gets the tools its spec lists (T1); an agent with rag = true first gets the corpus passages for
# the question (R1, R2), and its answer ends with their citations if the model did not cite them itself.
# A verified correction from the ReAct corrector (A1) is written to the learner's errors (M3).

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from contextlib import aclosing
from ..core.agent_spec import AgentSpec, spec_for
from ..core.errors import ErrorKind
from ..core.events import AgentRunFinished, Error, Event, HandoffEvent, Retrieved, RouteDecided, Token
from ..core.handoff import Handoff
from ..knowledge.retrieval import Retrieval
from ..memory.ledger import Ledger
from .context import TextBlock, build_context
from .executor import AgentRun
from .guards import GuardStop, TurnGuards
from .router import RouteDecision
from .session import Session

NO_FALLBACK: tuple[ErrorKind, ...] = ("connection", "turn_budget")
RETRIEVAL_TIMEOUT_S = 30


class Orchestrator:

    def __init__(self, session: Session, turn_id: str):
        self.session = session
        self.turn_id = turn_id
        self.guards = TurnGuards(session.config["orchestration"])
        self.runs: list[AgentRun] = []
        self.handoffs: list[Handoff] = []
        self.stop_kind: ErrorKind | None = None
        self.route: RouteDecision | None = None
        self.question = ""
        self.retrieval: Retrieval | None = None
        self.retrievals: dict[str, Retrieval] = {}

    @property
    def ok(self) -> bool:
        return bool(self.runs) and self.runs[-1].ok and self.stop_kind is None

    @property
    def answer_text(self) -> str:
        return self.runs[-1].text if self.runs else ""

    async def events(self, question: str) -> AsyncIterator[Event]:
        router = self.session.router
        self.question = router.strip_override(question) if router else question
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
        run.spec = self.session.prompts.resolve(self.session.persona.apply(run.spec))
        run.handoff = handoff
        run.tools = await asyncio.to_thread(self.session.knowledge.tools_for, run.spec)
        run.max_tool_rounds = self.session.config["tools"]["max_rounds"]
        self.runs.append(run)
        deadline, stop = self.guards.run_deadline(run.spec)
        extra = []
        retrieval = await self._retrieve(run, deadline) if run.spec.rag else None
        if retrieval is not None:
            yield Retrieved(turn_id=self.turn_id, run_id=run.run_id, **retrieval.as_event_fields())
            note = retrieval.context_note()
            if note:
                extra.append(TextBlock("rag", priority=1, max_share=0.35, text=note))
        async with aclosing(run.events(build_context(self.session, run.spec, handoff, extra))) as run_events:
            try:
                async for event in self._until(run_events, deadline):
                    yield event
            except asyncio.TimeoutError:
                yield self._stop(stop)
                yield AgentRunFinished(turn_id=self.turn_id, run_id=run.run_id, agent=run.spec.name, ok=False)
                return
        if run.ok and retrieval is not None:
            sources = _missing_citations(run.text, retrieval)
            if sources:
                run.text += sources
                yield Token(turn_id=self.turn_id, run_id=run.run_id, text=sources)

    async def _retrieve(self, run: AgentRun, deadline: float) -> Retrieval | None:
        if self.retrieval is None:
            profile = self.session.profile["content"] if self.session.profile else None
            seconds = min(RETRIEVAL_TIMEOUT_S, self.guards.seconds_left(deadline))
            try:
                self.retrieval = await asyncio.wait_for(asyncio.to_thread(
                    self.session.knowledge.retrieve, self.question, profile, run.spec.model), seconds)
            except (asyncio.TimeoutError, Exception):
                return None
        if self.retrieval is not None:
            self.retrievals[run.run_id] = self.retrieval
        return self.retrieval

    def record_errors(self) -> None:
        for run in self.runs:
            if run.ok and run.spec.strategy == "react":
                self._record_error(run)

    def _record_error(self, run: AgentRun) -> None:
        data = run.data or {}
        first_check = data.get("first_check") or {}
        sentence = first_check.get("sentence") or data.get("sentence")
        correction = data.get("correction")
        if not data.get("verified") or not correction or correction == sentence:
            return
        ledger = Ledger(self.session.repo.conn)
        try:
            if ledger.list("errors", sentence=sentence, correction=correction, resolved=0):
                return
            ledger.create("corrector", "errors", session_id=self.session.id if self.session.stored else None,
                          sentence=sentence, correction=correction,
                          error_type=first_check.get("error_type") or "unknown")
        except (sqlite3.Error, ValueError):
            return

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


def _missing_citations(answer: str, retrieval: Retrieval) -> str:
    if not retrieval.kept or retrieval.mode != "embeddings+model":
        return ""
    if any(citation in answer or chunk.section in answer for citation, chunk in zip(retrieval.citations,
                                                                                  retrieval.kept)):
        return ""
    return "\n\nSources: " + " ".join(retrieval.citations)
