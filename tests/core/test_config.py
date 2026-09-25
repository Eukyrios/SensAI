import pytest

from sensai.core import config as cfg
from sensai.core.config import DEFAULTS, load_config, resolve


def write(tmp_path, text):
    p = tmp_path / "config.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_defaults_when_no_file(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "missing.toml")
    assert load_config() == DEFAULTS


def test_repo_config_toml_loads():
    c = load_config()
    assert c["ollama"]["model"] and c["ollama"]["num_ctx"] > 0


def test_file_overrides_defaults(tmp_path):
    p = write(tmp_path, '[ollama]\nurl = "http://file"\ntimeout_s = 5\n')
    o = load_config(p)["ollama"]
    assert (o["url"], o["timeout_s"]) == ("http://file", 5)


def test_file_keeps_unset_defaults_and_adds_sections(tmp_path):
    c = load_config(write(tmp_path, '[ollama]\ntimeout_s = 1\n[extra]\nx = 1\n'))
    assert c["ollama"]["url"] == DEFAULTS["ollama"]["url"]
    assert c["extra"] == {"x": 1}


def test_explicit_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")


def test_resolve():
    assert resolve("data") == cfg.ROOT / "data"
    assert resolve("/abs") == cfg.Path("/abs")
