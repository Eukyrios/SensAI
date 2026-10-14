# Layered configuration: defaults < config.toml < SENSAI_<SECTION>__<KEY> env vars < CLI arguments.

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.toml"
ENV_PREFIX = "SENSAI_"

DEFAULTS: dict[str, dict[str, Any]] = {
    "ollama": {"url": "http://localhost:11434", "timeout_s": 60},
    "agents": {"dir": "agents", "default": "sensei", "model_override": ""},
    "storage": {"db_path": "data/sensai.db"},
    "memory": {"max_history_turns": 20, "profile": "default"},
    "guardrails": {"enabled": True},
    "logging": {"level": "INFO", "path": "logs/sensai.jsonl"},
}


def load_toml(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path | None = None,
                env: Mapping[str, str] | None = None,
                cli: Mapping[str, Mapping[str, Any]] | None = None) -> dict:
    config = copy.deepcopy(DEFAULTS)
    if path is not None:
        _merge(config, load_toml(path))
    elif CONFIG_PATH.exists():
        _merge(config, load_toml(CONFIG_PATH))
    _merge(config, _from_env(os.environ if env is None else env, config))
    if cli:
        _merge(config, {s: {k: v for k, v in keys.items() if v is not None} for s, keys in cli.items()})
    return config


def _merge(base: dict, override: Mapping) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _from_env(env: Mapping[str, str], config: dict) -> dict:
    out: dict[str, dict[str, Any]] = {}
    for name, raw in env.items():
        if not name.startswith(ENV_PREFIX) or "__" not in name:
            continue
        section, key = name[len(ENV_PREFIX):].lower().split("__", 1)
        current = config.get(section, {}).get(key)
        out.setdefault(section, {})[key] = _coerce(raw, current)
    return out


def _coerce(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw
