import dataclasses

import pytest

from sensai.core import events as ev
from sensai.core.agent_spec import AgentSpec, SpecError, load_spec, spec_for
from sensai.core.config import resolve
from sensai.core.handoff import Handoff
from sensai.core.ids import new_id
from tests.fakes import make_config

SENSEI = resolve("agents/sensei.toml")


def test_new_id_unique():
    assert new_id() != new_id() and len(new_id()) == 32


def all_events(turn):
    h = Handoff("sensei", "corrector", "grammar", {"sentence": "x"}, hop=1, run_id="r1")
    return [
        ev.TurnStarted(turn_id=turn, session_id="s", user_input="hi"),
        ev.AgentRunStarted(turn_id=turn, run_id="r1", agent="sensei"),
        ev.Thinking(turn_id=turn, run_id="r1", text="hmm"),
        ev.Token(turn_id=turn, run_id="r1", text="a"),
        ev.StructuredOutput(turn_id=turn, run_id="r1", data={"ok": True}),
        ev.ToolCall(turn_id=turn, run_id="r1", name="kanji", args={"k": "日"}),
        ev.ToolResult(turn_id=turn, run_id="r1", name="kanji", result="sun"),
        ev.HandoffEvent(turn_id=turn, handoff=h),
        ev.GuardrailTriggered(turn_id=turn, side="input", rule="pii", action="redact"),
        ev.Error(turn_id=turn, kind="connection", message="down"),
        ev.AgentRunFinished(turn_id=turn, run_id="r1", agent="sensei", ok=True),
        ev.TurnFinished(turn_id=turn, model="m", prompt_eval_count=10, eval_count=5),
    ]


def test_events_are_frozen_and_timestamped():
    for e in all_events("t"):
        assert e.turn_id == "t" and e.ts > 0
        with pytest.raises(dataclasses.FrozenInstanceError):
            e.turn_id = "other"


def test_events_dispatch_with_match():
    def render(e: ev.Event) -> str:
        match e:
            case ev.Token(text=t):
                return t
            case ev.Error(kind=k):
                return f"!{k}"
            case _:
                return ""
    assert "".join(render(e) for e in all_events("t")) == "a!connection"


def test_handoff_requires_dict_payload():
    with pytest.raises(TypeError):
        Handoff("a", "b", "r", payload="free text", hop=0, run_id="r")


def test_load_spec_ok():
    spec = load_spec(SENSEI)
    assert isinstance(spec, AgentSpec)
    assert spec.name == "sensei" and spec.model and spec.num_ctx > 0 and spec.tools == ()


@pytest.mark.parametrize("drop", ["model", "num_ctx", "name"])
def test_load_spec_missing_required(tmp_path, drop):
    lines = [l for l in SENSEI.read_text(encoding="utf-8").splitlines(True) if not l.startswith(f"{drop} ")]
    p = tmp_path / "bad.toml"
    p.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(SpecError, match=drop):
        load_spec(p)


def test_load_spec_unknown_field(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text('name="x"\nmodel="m"\nnum_ctx=1\ncolor="red"\n', encoding="utf-8")
    with pytest.raises(SpecError, match="color"):
        load_spec(p)


def test_spec_for_applies_model_override():
    assert spec_for("sensei", make_config()).model == load_spec(SENSEI).model
    assert spec_for("sensei", make_config(agents={"model_override": "other"})).model == "other"


def test_spec_for_unknown_agent():
    with pytest.raises(SpecError, match="unknown agent"):
        spec_for("nobody", make_config())


def test_load_spec_rejects_an_invalid_output_schema(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text('name="x"\nmodel="m"\nnum_ctx=1\n[output_schema]\ntype="not-a-type"\n', encoding="utf-8")
    with pytest.raises(SpecError, match="invalid output_schema"):
        load_spec(p)


def test_load_spec_accepts_a_valid_output_schema(tmp_path):
    p = tmp_path / "ok.toml"
    p.write_text('name="x"\nmodel="m"\nnum_ctx=1\n[output_schema]\ntype="object"\n', encoding="utf-8")
    assert load_spec(p).output_schema == {"type": "object"}
