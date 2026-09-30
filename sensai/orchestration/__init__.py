"""Turn loop and agent coordination, owned by A.

    session.py       Session: active agent, in-session history, model client    (layer 2)
    turn.py          run_turn(): guardrails -> context -> model -> events       (layer 2)
    executor.py      runs one AgentSpec, ReAct strategy for the corrector       (layer 4)
    orchestrator.py  handoffs and guards: max hops, cycles, budget, timeouts    (layer 4)
    router.py        rules -> embeddings -> classifier cascade                  (layer 5)
"""
