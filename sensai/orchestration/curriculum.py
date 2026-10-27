# Curriculum pass: after the learner has seen the whole answer, the curriculum agent reads the turn and
# updates the srs_cards itself by calling its tools, one step at a time (prompted tool calls, each step a
# validated structured reply). It is best effort: any failure is swallowed and never reaches the learner.

import sqlite3
from contextlib import aclosing
from dataclasses import replace
from datetime import date

from ..core.agent_spec import AgentSpec, spec_for
from ..core.messages import Message
from ..knowledge.tools import ToolRegistry
from ..memory import srs
from ..memory.ledger import Ledger
from .context import build_context
from .executor import AgentRun
from .session import Session

AGENT = srs.OWNER
DONE = "done"


async def run_curriculum(session: Session, turn_id: str, today: date | None = None) -> None:
    settings = session.config["curriculum"]
    if not settings["enabled"] or not session.history:
        return
    ledger = Ledger(session.repo.conn)
    try:
        spec = spec_for(AGENT, session.config)
        registry = _registry(ledger, spec, today or date.today(), session.history)
        spec = replace(spec, system_prompt=f"{spec.system_prompt}\n\n{registry.get_prompt_instructions()}")
        await _tool_loop(session, turn_id, spec, registry, ledger, settings["max_steps"])
    except Exception:
        return


async def _tool_loop(session: Session, turn_id: str, spec: AgentSpec, registry: ToolRegistry,
                     ledger: Ledger, max_steps: int) -> None:
    messages = build_context(session, spec) + [_request(ledger)]
    for _ in range(max_steps):
        run = AgentRun(session.client, spec, turn_id)
        async with aclosing(run.events(messages)) as events:
            async for _event in events:
                pass
        if not run.ok or run.data["tool"] == DONE:
            return
        result = registry.execute(run.data["tool"], run.data["args"])
        messages = messages + [{"role": "assistant", "content": run.text},
                               {"role": "user", "content": f"Tool result: {result}\n"
                                                           f"Call another tool, or reply {DONE}."}]


def _registry(ledger: Ledger, spec: AgentSpec, today: date, history: list[Message]) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in srs.tools(ledger, today, _exchange_text(history), _was_quizzed(history)):
        if tool.name in spec.tools:
            registry.register(tool)
    return registry


def _exchange_text(history: list[Message]) -> str:
    return " ".join(message["content"] for message in history[-2:])


def _was_quizzed(history: list[Message]) -> bool:
    """True when the tutor's message before the learner's last one asked a question."""
    if len(history) < 3 or history[-3]["role"] != "assistant":
        return False
    return "?" in history[-3]["content"] or "？" in history[-3]["content"]


def _request(ledger: Ledger) -> Message:
    known = _known_cards(ledger.list("srs_cards"))
    return {"role": "user", "content": f"Known cards:\n{known}\n\nUse your tools for the exchange above."}


def _known_cards(cards: list[sqlite3.Row]) -> str:
    return "\n".join(f"- {card['front']} = {card['back']}" for card in cards) or "(none)"
