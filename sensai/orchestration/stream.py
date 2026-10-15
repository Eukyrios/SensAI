# Thread bridge: runs the blocking Ollama stream in a thread and hands its pieces to asyncio, so a
# reply can be stopped mid-stream without blocking the event loop.

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from functools import partial

from ..core.agent_spec import AgentSpec
from ..core.messages import Message
from ..llm.ollama import ChatStats, Thought


async def stream_tokens(client, spec: AgentSpec, messages: list[Message],
                        format: dict | str | None = None) -> AsyncIterator[str | Thought | ChatStats]:
    options = {"format": format} if format is not None else {}
    open_stream = partial(client.chat_stream, spec.model, messages, num_ctx=spec.num_ctx,
                          temperature=spec.temperature, think=spec.think, **options)
    stream = _ThreadedStream(asyncio.get_running_loop())
    stream.start(open_stream)
    try:
        async for item in stream.items():
            yield item
    finally:
        stream.stop()


class _StreamFailed:
    def __init__(self, error: BaseException):
        self.error = error


class _StreamEnded:
    def __init__(self, stats: ChatStats | None):
        self.stats = stats


class _ThreadedStream:

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue()
        self.stop_requested = threading.Event()

    def start(self, open_stream: Callable[[], Iterator[str | Thought]]) -> None:
        threading.Thread(target=self._produce, args=(open_stream,), name="ollama-stream",
                         daemon=True).start()

    def stop(self) -> None:
        self.stop_requested.set()

    async def items(self) -> AsyncIterator[str | Thought | ChatStats]:
        while True:
            item = await self.queue.get()
            if isinstance(item, _StreamEnded):
                if item.stats is not None:
                    yield item.stats
                return
            if isinstance(item, _StreamFailed):
                raise item.error
            yield item

    def _produce(self, open_stream: Callable[[], Iterator[str | Thought]]) -> None:
        source = open_stream()
        try:
            self._forward(source)
        except BaseException as error:
            self._hand_over(_StreamFailed(error))
        finally:
            source.close()

    def _forward(self, source: Iterator[str | Thought]) -> None:
        while not self.stop_requested.is_set():
            try:
                piece = next(source)
            except StopIteration as end:
                self._hand_over(_StreamEnded(end.value))
                return
            self._hand_over(piece)

    def _hand_over(self, item) -> None:
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, item)
        except RuntimeError:
            pass
