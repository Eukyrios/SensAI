# Server-sent events: each event of the stream becomes one SSE message, named after its class,
# with the event's fields as JSON data. TurnFinished also carries the turn's metrics and trace.

import json
from collections.abc import AsyncIterator
from dataclasses import asdict

from ...core.events import Event, TurnFinished
from ...trust.metrics import compute_turn_metrics
from ...trust.tracing import TurnTracer


def event_message(event: Event, tracer: TurnTracer | None = None) -> str:
    payload = asdict(event)
    if isinstance(event, TurnFinished):
        payload["metrics"] = _metrics(event)
        if tracer is not None:
            payload["trace"] = tracer.format_trace()
    return f"event: {type(event).__name__}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def sse_stream(events: AsyncIterator[Event]) -> AsyncIterator[str]:
    tracer = TurnTracer()
    async for event in events:
        tracer.process_event(event)
        yield event_message(event, tracer)


def _metrics(event: TurnFinished) -> dict:
    metrics = compute_turn_metrics(event)
    return {"total_tokens": metrics.total_tokens, "tokens_per_sec": round(metrics.tokens_per_sec, 1),
            "cost_usd": round(metrics.estimated_cost_usd, 6), "duration_s": round(metrics.duration_s, 2)}
