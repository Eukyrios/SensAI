import pytest

from sensai.core.agent_spec import AgentSpec, SpecError, load_spec, spec_for
from sensai.core.config import resolve
from tests.fakes import make_config

SENSEI = resolve("agents/sensei.toml")


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
