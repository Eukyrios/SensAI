# Layered configuration: defaults < config.toml < SENSAI_<SECTION>__<KEY> env vars < CLI arguments.
# Nested tables take one more __ per level: SENSAI_MEMORY__SUMMARY__TRIGGER.

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
    "agents": {"dir": "agents", "default": "sensei", "fallback": "sensei", "model_override": ""},
    "orchestration": {"max_hops": 3, "turn_budget_s": 180},
    "router": {"enabled": True, "routes": "routing.toml", "embed_model": "bge-m3", "embed_on_cpu": True,
               "embedding_threshold": 0.55, "embedding_margin": 0.05, "classifier": "classifier",
               "confidence_floor": 0.5, "internal_agents": ["curriculum", "classifier", "demo/*"],
               "debug": False},
    "personas": {"dir": "personas", "default": "teacher"},
    "storage": {"db_path": "data/sensai.db"},
    "memory": {"max_history_turns": 20, "profile": "default",
               "summary": {"enabled": True, "trigger": 0.7, "keep_recent": 6}},
    "curriculum": {"enabled": True, "max_steps": 8},
    "tools": {"enabled": True, "max_rounds": 3, "disabled": []},
    "mcp": {"enabled": True, "servers": []},
    "rag": {"enabled": True, "corpus_dir": "corpus", "embed_model": "bge-m3", "embed_on_cpu": True,
            "index_path": "data/rag_index.json", "default_level": "", "candidate_k": 5, "min_similarity": 0.4,
            "rerank": True, "model_rerank": True, "relevance_threshold": 0.6, "hybrid_threshold": 0.5,
            "lexical_threshold": 0.4, "top_k": 2},
    "guardrails": {"enabled": True, "block_prompt_injection": True, "block_out_of_scope": True,
                   "redact_pii_input": True, "redact_pii_output": True},
    "eval": {"enabled": True, "judge_model": "", "max_context_chars": 6000},
    "logging": {"level": "INFO", "path": "logs/sensai.jsonl"},
    "web": {"host": "127.0.0.1", "port": 8000, "allow_origins": ["http://localhost:5173"]},
}


def load_toml(path: str | Path) -> dict:
    with open(path, "rb") as toml_file:
        return tomllib.load(toml_file)


def resolve(path: str | Path) -> Path:
    given_path = Path(path)
    return given_path if given_path.is_absolute() else ROOT / given_path


def load_config(path: str | Path | None = None,
                env: Mapping[str, str] | None = None,
                cli: Mapping[str, Mapping[str, Any]] | None = None) -> dict:
    config = copy.deepcopy(DEFAULTS)
    if path is not None:
        _merge_into(config, load_toml(path))
    elif CONFIG_PATH.exists():
        _merge_into(config, load_toml(CONFIG_PATH))
    _merge_into(config, _values_from_env(os.environ if env is None else env, config))
    if cli:
        _merge_into(config, _given_cli_values(cli))
    return config


def _given_cli_values(cli: Mapping[str, Mapping[str, Any]]) -> dict:
    return {section: {key: value for key, value in values.items() if value is not None}
            for section, values in cli.items()}


def _merge_into(base: dict, override: Mapping) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _merge_into(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _values_from_env(env: Mapping[str, str], config: dict) -> dict:
    out: dict[str, Any] = {}
    for name, raw in env.items():
        if not name.startswith(ENV_PREFIX) or "__" not in name:
            continue
        path = name[len(ENV_PREFIX):].lower().split("__")
        _set_path(out, path, _convert_env_value(raw, _get_path(config, path)))
    return out


def _get_path(config: dict, path: list[str]) -> Any:
    value: Any = config
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def _set_path(out: dict, path: list[str], value: Any) -> None:
    for key in path[:-1]:
        out = out.setdefault(key, {})
    out[path[-1]] = value


def _convert_env_value(raw: str, current: Any) -> Any:
    if isinstance(current, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    return raw
