# Tests for the orchestrator: a checker hands the turn to the sensei with its findings as payload.

import json

from sensai.core import events as ev
from sensai.core.config import resolve
from sensai.llm.ollama import ChatStats, OllamaError
from sensai.orchestration.session import Session
from sensai.orchestration.turn import run_turn
from tests.fakes import ScriptedClient, make_config

FINDINGS = {"sentence": "私は学校を行きます", "is_correct": False, "correction": "私は学校に行きます",
            "error_type": "particle", "rule": "Use に for the destination of a movement verb."}
WRONG = json.dumps({**FINDINGS, "handoff": {"to": "sensei", "reason": "particle mistake"}}, ensure_ascii=False)
RIGHT = json.dumps({**FINDINGS, "is_correct": True, "error_type": "none", "handoff": None}, ensure_ascii=False)
EXPLANATION = "Use に (ni) here: 学校に行きます."


class FailingSecondCall(ScriptedClient):

    def chat_stream(self, model, messages, **options):
        if self.calls:
            raise OllamaError("connection", "down")
        yield from super().chat_stream(model, messages, **options)


def checker_session(client):
    return Session(make_config(agents={"default": "checker"}), client=client)


async def collect(session, text="私は学校を行きます"):
    return [event async for event in run_turn(session, text)]


def kinds(events):
    return [type(event).__name__ for event in events]


async def test_handoff_runs_the_second_agent_under_the_first():
    events = await collect(checker_session(ScriptedClient(WRONG, EXPLANATION)))
    without_tokens = [kind for kind in kinds(events) if kind != "Token"]
    assert without_tokens == ["TurnStarted", "AgentRunStarted", "StructuredOutput", "AgentRunFinished",
                              "HandoffEvent", "AgentRunStarted", "AgentRunFinished", "TurnFinished"]
    assert "".join(event.text for event in events if isinstance(event, ev.Token)) == EXPLANATION
    checker_run, sensei_run = [event for event in events if isinstance(event, ev.AgentRunStarted)]
    assert (checker_run.agent, sensei_run.agent) == ("checker", "sensei")
    assert sensei_run.parent_run_id == checker_run.run_id


async def test_handoff_carries_the_validated_findings():
    events = await collect(checker_session(ScriptedClient(WRONG, EXPLANATION)))
    handoff = next(event for event in events if isinstance(event, ev.HandoffEvent)).handoff
    assert (handoff.from_agent, handoff.to_agent, handoff.reason) == ("checker", "sensei", "particle mistake")
    assert handoff.payload == FINDINGS and handoff.hop == 1
    assert handoff.run_id == events[1].run_id and handoff.parent_run_id is None


async def test_receiving_agent_gets_its_prompt_the_note_and_the_question():
    session = checker_session(ScriptedClient(WRONG, EXPLANATION))
    await collect(session)
    model, messages, options = session.client.calls[1]
    assert model == "qwen3.5:9b" and "format" not in options
    system = messages[0]["content"]
    assert system.startswith("You are Sensai") and "checker agent looked at this turn first" in system
    assert "に for the destination" in system
    assert messages[1:] == [{"role": "user", "content": "私は学校を行きます"}]


async def test_the_last_agent_answer_is_saved():
    session = checker_session(ScriptedClient(WRONG, EXPLANATION))
    await collect(session)
    assert session.history == [{"role": "user", "content": "私は学校を行きます"},
                               {"role": "assistant", "content": EXPLANATION}]


async def test_no_handoff_ends_the_turn_with_the_first_agent(tmp_path):
    for name in ("checker", "sensei"):
        spec = resolve(f"agents/{name}.toml").read_text(encoding="utf-8")
        (tmp_path / f"{name}.toml").write_text(spec.replace("handoff_required = true\n", ""), encoding="utf-8")
    session = Session(make_config(agents={"default": "checker", "dir": str(tmp_path)}), client=ScriptedClient(RIGHT))
    events = await collect(session)
    assert kinds(events).count("AgentRunStarted") == 1 and "HandoffEvent" not in kinds(events)
    assert session.history[-1] == {"role": "assistant", "content": RIGHT}


async def test_stats_and_model_cover_every_run():
    stats = ChatStats(prompt_eval_count=10, eval_count=4, eval_duration_ns=100)
    finished = (await collect(checker_session(ScriptedClient(WRONG, EXPLANATION, stats=stats))))[-1]
    assert (finished.prompt_eval_count, finished.eval_count, finished.eval_duration_ns) == (20, 8, 200)
    assert finished.model == "qwen3.5:9b"


async def test_failure_after_a_handoff_rolls_the_question_back():
    session = checker_session(FailingSecondCall(WRONG))
    events = await collect(session)
    assert kinds(events)[-3:] == ["Error", "AgentRunFinished", "TurnFinished"]
    assert events[-2].agent == "sensei" and events[-2].ok is False
    assert session.history == []
