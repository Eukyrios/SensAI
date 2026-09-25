"""Shared contracts, owned by A.

Every other package codes against these types. Change them only after telling the team.
    messages.py    chat message format sent to the model
    errors.py      error categories (ErrorKind)
    config.py      config loader: defaults < config.toml (env and CLI levels in layer 1)
    events.py      the event stream that surfaces subscribe to                 (layer 1)
    handoff.py     the envelope passed between agents                          (layer 1)
    ids.py         id generation for turns, runs and sessions                  (layer 1)
    agent_spec.py  AgentSpec, loaded from agents/<name>.toml                   (layer 1)
"""
