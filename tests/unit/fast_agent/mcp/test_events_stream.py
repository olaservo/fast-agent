from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from mcp.shared.exceptions import MCPError
from mcp.shared.subscriptions import SUBSCRIPTION_ID_META_KEY

from fast_agent.mcp.events_extension import (
    EventStreamActive,
    EventStreamFrame,
    EventStreamHeartbeat,
)
from fast_agent.mcp.events_stream import EventStreamRouter

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from mcp_types import RequestId


async def _notify(router: EventStreamRouter, method: str, frame: EventStreamFrame) -> None:
    handlers = {binding.method: binding.handler for binding in router.notifications()}
    await handlers[f"notifications/events/{method}"](frame)


def _active(subscription_id: RequestId) -> EventStreamActive:
    return EventStreamActive(cursor="c0", meta={SUBSCRIPTION_ID_META_KEY: subscription_id})


def _heartbeat(subscription_id: RequestId, cursor: str) -> EventStreamHeartbeat:
    return EventStreamHeartbeat(cursor=cursor, meta={SUBSCRIPTION_ID_META_KEY: subscription_id})


@pytest.mark.asyncio
async def test_streams_open_one_at_a_time_and_each_receives_its_own_messages() -> None:
    router = EventStreamRouter()
    release = asyncio.Event()
    sent: list[str] = []

    def request(label: str) -> Callable[[], Awaitable[str]]:
        async def send() -> str:
            sent.append(label)
            await release.wait()
            return label

        return send

    first: list[EventStreamFrame] = []
    second: list[EventStreamFrame] = []
    one = asyncio.create_task(router.run(request("one"), first.append))
    two = asyncio.create_task(router.run(request("two"), second.append))
    for _ in range(5):
        await asyncio.sleep(0)
    # The second request waits until the first stream has named itself.
    assert sent == ["one"]

    await _notify(router, "active", _active(7))
    while sent != ["one", "two"]:
        await asyncio.sleep(0)
    await _notify(router, "active", _active("stream-8"))

    await _notify(router, "heartbeat", _heartbeat("7", "c1"))
    await _notify(router, "heartbeat", _heartbeat("stream-8", "c2"))
    await _notify(router, "heartbeat", _heartbeat(99, "c3"))
    await _notify(router, "heartbeat", EventStreamHeartbeat(cursor="c4"))

    assert first == [_active(7), _heartbeat("7", "c1")]
    assert second == [_active("stream-8"), _heartbeat("stream-8", "c2")]

    release.set()
    assert (await one, await two) == ("one", "two")
    await _notify(router, "heartbeat", _heartbeat(7, "c5"))
    assert len(first) == 2


@pytest.mark.asyncio
async def test_an_unconfirmed_stream_times_out_and_a_rejected_one_raises() -> None:
    router = EventStreamRouter(open_timeout_seconds=0.01)
    frames: list[EventStreamFrame] = []
    cancelled = asyncio.Event()

    async def never_answers() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with pytest.raises(TimeoutError):
        await router.run(never_answers, frames.append)
    assert cancelled.is_set()

    async def rejects() -> None:
        raise MCPError(-32602, "InvalidParams")

    with pytest.raises(MCPError):
        await router.run(rejects, frames.append)
    assert frames == []
