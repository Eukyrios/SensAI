"""Shared contracts, owned by A.

Every other package codes against these types. Change them only after telling the team.
    messages.py    chat message format sent to the model
    errors.py      error categories (ErrorKind)
    config.py      layered config: defaults < config.toml < env < CLI
    events.py      the event stream that surfaces subscribe to
    handoff.py     the envelope passed between agents
    ids.py         id generation for turns, runs and sessions
    agent_spec.py  AgentSpec, loaded from agents/<name>.toml
"""
