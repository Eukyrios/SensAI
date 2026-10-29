# Server-sent events: each event of the stream becomes one SSE message, named after its class,
# with the event's fields as JSON data.

import json
from collections.abc import AsyncIterator
from dataclasses import asdict

from ...core.events import Event


def event_message(event: Event) -> str:
    data = json.dumps(asdict(event), ensure_ascii=False)
    return f"event: {type(event).__name__}\ndata: {data}\n\n"


async def sse_stream(events: AsyncIterator[Event]) -> AsyncIterator[str]:
    async for event in events:
        yield event_message(event)
