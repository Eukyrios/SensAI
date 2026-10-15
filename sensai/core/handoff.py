# Handoff: the envelope one agent passes to another; the payload is JSON data, never prose.
# An agent asks for a handoff through the `handoff` field that handoff_schema() adds to its output.

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


def handoff_schema(targets: tuple[str, ...]) -> dict:
    return {
        "type": ["object", "null"],
        "description": "Pass this turn to another agent, with your other fields as its input; "
                       "null to end the turn with your own reply.",
        "properties": {"to": {"type": "string", "enum": list(targets)},
                       "reason": {"type": "string", "description": "Why, in a few words."}},
        "required": ["to", "reason"],
        "additionalProperties": False,
    }
