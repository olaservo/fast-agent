"""Client-side routing for MCP Events push delivery (SEP-3415 ``events/stream``)."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mcp.client.extension import ClientExtension, NotificationBinding
from mcp.shared.dispatcher import as_request_id, coerce_request_id
from mcp.shared.subscriptions import SUBSCRIPTION_ID_META_KEY

from fast_agent.mcp.events_extension import (
    EVENT_STREAM_NOTIFICATIONS,
    EVENTS_EXTENSION,
    EventFrameHandler,
    EventStreamFrame,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mcp_types import RequestId

STREAM_OPEN_TIMEOUT_SECONDS = 30.0


@dataclass
class _Stream:
    on_frame: EventFrameHandler
    bound: asyncio.Future[None]
    subscription_id: RequestId | None = None


class EventStreamRouter(ClientExtension):
    """Advertise the Events extension and hand each stream its notifications.

    A ``notifications/events/*`` message names its stream by the id of the
    ``events/stream`` request. ``ClientSession.send_request`` does not report
    the id it assigns, so streams open one at a time and the first message
    with an unknown id belongs to the stream that is opening.

    The SDK delivers each notification method through its own queue, so two
    messages of different kinds can arrive here out of wire order. A cursor
    that moves backwards replays events on reconnect, and the subscription
    manager drops those by ``eventId``.
    """

    identifier = EVENTS_EXTENSION

    def __init__(self, *, open_timeout_seconds: float = STREAM_OPEN_TIMEOUT_SECONDS) -> None:
        self._open_timeout_seconds = open_timeout_seconds
        self._streams: dict[RequestId, _Stream] = {}
        self._opening: _Stream | None = None
        self._open_lock = asyncio.Lock()

    def notifications(self) -> list[NotificationBinding[Any]]:
        return [
            NotificationBinding(method=method, params_type=frame_type, handler=self._receive)
            for method, frame_type in EVENT_STREAM_NOTIFICATIONS.items()
        ]

    async def _receive(self, frame: EventStreamFrame) -> None:
        raw_id = as_request_id((frame.meta or {}).get(SUBSCRIPTION_ID_META_KEY))
        if raw_id is None:
            return
        subscription_id = coerce_request_id(raw_id)
        stream = self._streams.get(subscription_id)
        if stream is None:
            stream = self._opening
            if stream is None:
                return
            self._opening = None
            stream.subscription_id = subscription_id
            self._streams[subscription_id] = stream
            stream.bound.set_result(None)
        stream.on_frame(frame)

    async def run[T](self, send: Callable[[], Awaitable[T]], on_frame: EventFrameHandler) -> T:
        """Hold one stream open and return the result that closes it.

        ``send`` issues the ``events/stream`` request. Cancel this call to end
        the stream. Raises what the request raises, and ``TimeoutError`` when
        the server neither answers nor sends a message within the open timeout.
        """
        stream = _Stream(on_frame, asyncio.get_running_loop().create_future())
        request: asyncio.Future[T] | None = None
        try:
            async with self._open_lock:
                self._opening = stream
                request = asyncio.ensure_future(send())
                try:
                    done, _ = await asyncio.wait(
                        {request, stream.bound},
                        timeout=self._open_timeout_seconds,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    stream.bound.cancel()
                    if self._opening is stream:
                        self._opening = None
                if not done:
                    raise TimeoutError("The server did not confirm the event stream")
            return await request
        finally:
            if request is not None and not request.done():
                request.cancel()
                with suppress(asyncio.CancelledError):
                    await request
            if stream.subscription_id is not None:
                self._streams.pop(stream.subscription_id, None)
