# Tests for personas: the persona files, applying one to an agent, and adding one without code.

import pytest

from sensai.core.agent_spec import AgentSpec, SpecError
from sensai.core.persona import Persona, load_persona, persona_for, persona_names
from tests.fakes import make_config

SENSEI = AgentSpec(name="sensei", model="m", num_ctx=1, system_prompt="You teach Japanese.")
CASUAL = Persona("casual", "Casual partner", "Friendly, plain forms.", "Everyday conversation.")


def test_the_three_personas_ship_with_the_project():
    config = make_config()
    assert persona_names(config) == ["casual", "strict", "teacher"]
    assert config["personas"]["default"] == "teacher"
    assert all(persona_for(name, config).tone for name in persona_names(config))


def test_a_persona_is_appended_to_a_prose_agent():
    prompt = CASUAL.apply(SENSEI).system_prompt
    assert prompt.startswith("You teach Japanese.\n\nPersona: casual.\nTone: Friendly, plain forms.\n"
                             "Scope: Everyday conversation.\n")
    assert prompt.endswith("keep every rule above, including the format of the Japanese.")


def test_agents_with_structured_output_keep_their_prompt():
    checker = AgentSpec(name="checker", model="m", num_ctx=1, system_prompt="Check.", output_schema={"type": "object"})
    assert CASUAL.apply(checker) is checker


def test_each_persona_gives_the_prompt_its_own_hash():
    hashes = {persona_for(name, make_config()).apply(SENSEI).prompt_hash for name in ("casual", "strict", "teacher")}
    assert len(hashes) == 3


def test_a_new_file_is_a_new_persona(tmp_path):
    (tmp_path / "pirate.toml").write_text('name = "pirate"\ndescription = "Arr"\ntone = "Talk like a pirate."\n'
                                          'scope = "Japanese for sailors."\n', encoding="utf-8")
    config = make_config(personas={"dir": str(tmp_path)})
    assert persona_names(config) == ["pirate"]
    assert "Talk like a pirate." in persona_for("pirate", config).apply(SENSEI).system_prompt


@pytest.mark.parametrize("text, problem", [
    ('name = "x"\ndescription = "d"\ntone = "t"\n', "missing required field.*scope"),
    ('name = "x"\ndescription = "d"\ntone = "t"\nscope = "s"\ncolor = "red"\n', "color"),
])
def test_bad_persona_files_are_rejected(tmp_path, text, problem):
    path = tmp_path / "bad.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(SpecError, match=problem):
        load_persona(path)


def test_unknown_persona():
    with pytest.raises(SpecError, match="unknown persona 'nobody'"):
        persona_for("nobody", make_config())


@pytest.mark.parametrize("name", ["../config", "teacher/../../config", "/etc/passwd", "a b", ""])
def test_a_persona_name_cannot_leave_the_personas_folder(name):
    with pytest.raises(SpecError, match="invalid persona name"):
        persona_for(name, make_config())
