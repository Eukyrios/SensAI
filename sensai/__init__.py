"""Sensai: a local, multi-agent Japanese tutor running on Ollama.

Package map (one domain per member, see README "Architecture"):
    core/           shared contracts: events, agent specs, handoffs, config   (A)
    llm/            Ollama HTTP transport                                      (C)
    orchestration/  turn loop, agent executor, orchestrator, router            (A)
    memory/         SQLite storage, sessions, ledger, context budgeting        (B)
    knowledge/      tools, MCP client, RAG pipeline                            (C)
    trust/          guardrails, judge, logging, metrics                        (D)
    surfaces/       CLI and web UI, which only consume the event stream        (D)
"""

__version__ = "0.1.0"
