# JSONL event logger: writes each event of the turn stream and passes it on.

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, AsyncGenerator

from ..core.events import Event


class JsonlLogger:
    def __init__(self, log_path: Path | str) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize_event(self, event: Event) -> dict[str, Any]:
        if is_dataclass(event):
            data = asdict(event)
        elif hasattr(event, "__dict__"):
            data = dict(event.__dict__)
        else:
            data = {"data": str(event)}

        data["event_type"] = event.__class__.__name__
        return data

    async def log_stream(self, stream: AsyncGenerator[Event, None]) -> AsyncGenerator[Event, None]:
        with open(self.log_path, "a", encoding="utf-8") as log_file:
            async for event in stream:
                record = self._serialize_event(event)
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
                yield event
