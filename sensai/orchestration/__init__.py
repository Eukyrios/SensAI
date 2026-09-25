"""Turn loop and agent coordination, owned by A.

    chat.py          base single-agent loop (B1), replaced by turn.py in layer 2
    turn.py          run_turn(): guardrails -> context -> model -> events       (layer 2)
    executor.py      runs one AgentSpec, ReAct strategy for the corrector       (layer 4)
    orchestrator.py  handoffs and guards: max hops, cycles, budget, timeouts    (layer 4)
    router.py        rules -> embeddings -> classifier cascade                  (layer 5)
"""
