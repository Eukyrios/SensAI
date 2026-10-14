# Handoff: the envelope one agent passes to another; the payload is JSON data, never prose.

from dataclasses import dataclass


@dataclass(frozen=True)
class Handoff:
    from_agent: str
    to_agent: str
    reason: str
    payload: dict
    hop: int
    run_id: str
    parent_run_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.payload, dict):
            raise TypeError("Handoff.payload must be a dict (validated JSON)")
