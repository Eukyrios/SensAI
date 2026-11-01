# Tests for the layered config loader.

import pytest

from sensai.core import config as cfg
from sensai.core.config import DEFAULTS, load_config, resolve


def write(tmp_path, text):
    p = tmp_path / "config.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_defaults_when_no_file(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "missing.toml")
    assert load_config(env={}) == DEFAULTS


def test_repo_config_toml_loads():
    assert load_config(env={})["agents"]["default"] == "sensei"


def test_file_overrides_defaults(tmp_path):
    p = write(tmp_path, '[ollama]\nurl = "http://file"\ntimeout_s = 5\n')
    o = load_config(p, env={})["ollama"]
    assert (o["url"], o["timeout_s"]) == ("http://file", 5)


def test_precedence_file_env_cli(tmp_path):
    p = write(tmp_path, '[ollama]\nurl = "http://file"\ntimeout_s = 5\n')
    env = {"SENSAI_OLLAMA__URL": "http://env"}
    assert load_config(p, env=env)["ollama"]["url"] == "http://env"

    c = load_config(p, env=env, cli={"ollama": {"url": "http://cli", "timeout_s": None}})
    assert (c["ollama"]["url"], c["ollama"]["timeout_s"]) == ("http://cli", 5)


@pytest.mark.parametrize("raw, key, expected", [
    ("false", "GUARDRAILS__ENABLED", False),
    ("yes", "GUARDRAILS__ENABLED", True),
    ("7", "MEMORY__MAX_HISTORY_TURNS", 7),
    ("x", "NEW__KEY", "x"),
])
def test_env_coercion(raw, key, expected):
    section, name = key.lower().split("__")
    assert load_config(env={f"SENSAI_{key}": raw})[section][name] == expected


def test_env_ignores_unrelated_names_and_coerces_floats():
    c = load_config(env={"OTHER": "x", "SENSAI_NOSEPARATOR": "x"})
    assert "noseparator" not in c and "other" not in c
    assert cfg._convert_env_value("1.5", 0.5) == 1.5


def test_file_keeps_unset_defaults_and_adds_sections(tmp_path):
    c = load_config(write(tmp_path, '[ollama]\ntimeout_s = 1\n[extra]\nx = 1\n'), env={})
    assert c["ollama"]["url"] == DEFAULTS["ollama"]["url"]
    assert c["extra"] == {"x": 1}


def test_explicit_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml", env={})


def test_resolve():
    assert resolve("agents") == cfg.ROOT / "agents"
    assert resolve("/abs") == cfg.Path("/abs")


def test_memory_profile_defaults_to_default():
    assert load_config(env={})["memory"]["profile"] == "default"


def test_env_reaches_nested_tables():
    config = load_config(env={"SENSAI_MEMORY__SUMMARY__TRIGGER": "0.02", "SENSAI_MEMORY__SUMMARY__ENABLED": "false"})
    assert config["memory"]["summary"]["trigger"] == 0.02
    assert config["memory"]["summary"]["enabled"] is False
    assert config["memory"]["summary"]["keep_recent"] == DEFAULTS["memory"]["summary"]["keep_recent"]
