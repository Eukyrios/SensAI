"""Configuration: built-in defaults < config.toml."""
import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:     # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.toml"

DEFAULTS: dict[str, dict[str, Any]] = {
    "ollama": {"url": "http://localhost:11434", "timeout_s": 60,
               "model": "qwen3.5:4b", "num_ctx": 8192, "temperature": 0.3, "think": False},
    "chat": {"system_prompt": "You are Sensai, a patient Japanese teacher. "
                              "Answer in the learner's language and give examples in Japanese."},
    "storage": {"db_path": "data/sensai.db"},
    "memory": {"max_history_turns": 20},
    "guardrails": {"enabled": True},
    "logging": {"level": "INFO", "path": "logs/sensai.jsonl"},
}


def load_toml(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path | None = None) -> dict:
    """Build the merged config. An explicit path must exist; the default config.toml is optional."""
    config = copy.deepcopy(DEFAULTS)
    if path is not None:
        _merge(config, load_toml(path))
    elif CONFIG_PATH.exists():
        _merge(config, load_toml(CONFIG_PATH))
    return config


def _merge(base: dict, override: Mapping) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
