from dataclasses import dataclass


@dataclass(frozen=True)
class Handoff:
    """Work passed from one agent to another. The payload is JSON data, never free prose."""
    from_agent: str
    to_agent: str
    reason: str
    payload: dict
    hop: int                    # counted by the orchestrator to enforce max hops
    run_id: str
    parent_run_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.payload, dict):
            raise TypeError("Handoff.payload must be a dict (validated JSON)")
