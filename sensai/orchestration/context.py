# Context of one agent run: its system prompt, the learner profile, the note of the handoff that
# brought the turn to this agent, then the recent conversation.

import json

from ..core.agent_spec import AgentSpec
from ..core.handoff import Handoff
from ..core.messages import Message
from .executor import system_prompt
from .session import Session


def build_context(session: Session, spec: AgentSpec, handoff: Handoff | None = None) -> list[Message]:
    max_turns = session.config["memory"]["max_history_turns"]
    profile = [session.profile] if session.profile else []
    note = [handoff_note(handoff)] if handoff is not None else []
    return ([{"role": "system", "content": system_prompt(spec)}]
            + profile
            + note
            + session.history[-max_turns * 2:])


def handoff_note(handoff: Handoff) -> Message:
    findings = json.dumps(handoff.payload, ensure_ascii=False)
    return {"role": "system",
            "content": f"The {handoff.from_agent} agent looked at this turn first ({handoff.reason}). "
                       f"Use its findings to answer the learner's last message: {findings}"}
