from __future__ import annotations

import pytest

from fast_agent.context import Context
from fast_agent.mcp.mcp_aggregator import MCPAggregator


class _BaseAggregator(MCPAggregator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialized = True

    async def validate_server(self, server_name: str) -> bool:
        return server_name in self.server_names


class _SubscribableAggregator(_BaseAggregator):
    """Aggregator whose servers advertise `resources.subscribe` and record dispatch calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls: list[tuple[str, str, str]] = []

    async def _server_supports_resource_subscribe(self, server_name: str) -> bool:
        del server_name
        return True

    async def _execute_on_server(
        self,
        server_name: str,
        operation_type: str,
        operation_name: str,
        method_name: str,
        method_args=None,
        error_factory=None,
        progress_callback=None,
    ):
        del operation_type, error_factory, progress_callback
        uri = str(method_args["uri"]) if method_args else ""
        self.calls.append((method_name, server_name, uri))
        # mcp `subscribe_resource` / `unsubscribe_resource` return EmptyResult.
        return None


def _make_aggregator(cls, *, connection_persistence: bool = False):
    return cls(
        server_names=["demo"],
        connection_persistence=connection_persistence,
        context=Context(),
    )


@pytest.mark.asyncio
async def test_subscribe_resource_dispatches_and_records() -> None:
    aggregator = _make_aggregator(_SubscribableAggregator)

    result = await aggregator.subscribe_resource("file:///demo.txt", "demo")

    assert aggregator.calls == [("subscribe_resource", "demo", "file:///demo.txt")]
    assert aggregator.get_subscriptions() == {"demo": {"file:///demo.txt"}}
    # Non-persistent connections cannot receive update notifications.
    assert result is False


@pytest.mark.asyncio
async def test_subscribe_returns_true_with_persistent_connection() -> None:
    aggregator = _make_aggregator(_SubscribableAggregator, connection_persistence=True)

    result = await aggregator.subscribe_resource("file:///demo.txt", "demo")

    assert result is True


@pytest.mark.asyncio
async def test_unsubscribe_resource_dispatches_and_clears() -> None:
    aggregator = _make_aggregator(_SubscribableAggregator)

    await aggregator.subscribe_resource("file:///demo.txt", "demo")
    await aggregator.unsubscribe_resource("file:///demo.txt", "demo")

    assert aggregator.calls[-1] == ("unsubscribe_resource", "demo", "file:///demo.txt")
    # The server key is dropped once its last subscription is removed.
    assert aggregator.get_subscriptions() == {}


@pytest.mark.asyncio
async def test_get_subscriptions_returns_a_copy() -> None:
    aggregator = _make_aggregator(_SubscribableAggregator)

    await aggregator.subscribe_resource("file:///demo.txt", "demo")
    snapshot = aggregator.get_subscriptions()
    snapshot["demo"].add("file:///mutated.txt")

    # Mutating the returned copy must not affect internal state.
    assert aggregator.get_subscriptions() == {"demo": {"file:///demo.txt"}}


@pytest.mark.asyncio
async def test_subscribe_unknown_server_raises() -> None:
    aggregator = _make_aggregator(_SubscribableAggregator)

    with pytest.raises(ValueError, match="not found"):
        await aggregator.subscribe_resource("file:///demo.txt", "missing")


@pytest.mark.asyncio
async def test_subscribe_without_capability_raises() -> None:
    class _NoSubscribeAggregator(_SubscribableAggregator):
        async def _server_supports_resource_subscribe(self, server_name: str) -> bool:
            del server_name
            return False

    aggregator = _make_aggregator(_NoSubscribeAggregator)

    with pytest.raises(ValueError, match="does not support resource subscriptions"):
        await aggregator.subscribe_resource("file:///demo.txt", "demo")

    assert aggregator.calls == []


@pytest.mark.asyncio
async def test_handle_resource_updated_surfaces_and_invokes_hook(monkeypatch) -> None:
    aggregator = _make_aggregator(_SubscribableAggregator)

    badges: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "fast_agent.ui.notification_tracker.add_resource_update",
        lambda server, uri: badges.append((server, uri)),
    )

    invalidations: list[tuple[str, str]] = []

    async def _record_invalidation(server_name: str, uri: str) -> None:
        invalidations.append((server_name, uri))

    monkeypatch.setattr(aggregator, "_on_resource_invalidated", _record_invalidation)

    await aggregator._handle_resource_updated("demo", "file:///demo.txt")

    assert badges == [("demo", "file:///demo.txt")]
    assert invalidations == [("demo", "file:///demo.txt")]
