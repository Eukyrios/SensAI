"""Safety, evaluation and observability, owned by D.

    guardrails.py  input scope/PII checks and output redaction                 (stub now, layer 6)
    logging.py     structured JSON logger subscribed to the event stream       (layer 1)
    metrics.py     tokens, latency, TTFT, cost per turn                        (layer 3)
    trace.py       per-turn, per-agent-run trace                               (layer 4)
    judge.py       claim-by-claim verification against retrieved chunks       (layer 5)
"""
