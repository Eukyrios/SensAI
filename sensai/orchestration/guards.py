# Guards of one turn: a maximum number of handoffs, no agent receiving the turn twice, a time budget
# for the whole turn and each agent's own timeout. They stop a crew that loops or hangs.

import time
from collections.abc import Callable
from dataclasses import dataclass

from ..core.agent_spec import AgentSpec
from ..core.errors import ErrorKind
from ..core.handoff import Handoff


@dataclass(frozen=True)
class GuardStop:
    kind: ErrorKind
    message: str


class TurnGuards:

    def __init__(self, settings: dict, clock: Callable[[], float] = time.monotonic):
        self.max_hops = settings["max_hops"]
        self.budget_s = settings["turn_budget_s"]
        self.clock = clock
        self.deadline = clock() + self.budget_s

    def stop_before(self, handoff: Handoff, agents_run: list[str]) -> GuardStop | None:
        if handoff.hop > self.max_hops:
            return GuardStop("max_hops", f"Stopped after {self.max_hops} handoffs in one turn "
                                         f"({handoff.from_agent} → {handoff.to_agent})")
        if handoff.to_agent in agents_run:
            return GuardStop("handoff_cycle", f"Stopped a handoff loop: {handoff.to_agent} already "
                                              f"ran in this turn ({' → '.join(agents_run)})")
        if self.clock() >= self.deadline:
            return self.budget_stop()
        return None

    def run_deadline(self, spec: AgentSpec) -> tuple[float, GuardStop]:
        agent_deadline = self.clock() + spec.timeout_s
        if agent_deadline < self.deadline:
            return agent_deadline, GuardStop("timeout", f"{spec.name} took longer than {spec.timeout_s}s")
        return self.deadline, self.budget_stop()

    def seconds_left(self, deadline: float) -> float:
        return max(deadline - self.clock(), 0.0)

    def budget_stop(self) -> GuardStop:
        return GuardStop("turn_budget", f"The turn went over its {self.budget_s}s budget")
