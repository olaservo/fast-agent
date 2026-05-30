from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from fast_agent.commands.context import CommandContext
from fast_agent.commands.handlers.resources import handle_resources_command
from fast_agent.interfaces import AgentProtocol


def _make_context(agent: object) -> CommandContext:
    provider = SimpleNamespace(_agent=lambda name: agent)
    return CommandContext(
        agent_provider=provider,  # type: ignore[arg-type]
        current_agent_name="demo-agent",
        io=SimpleNamespace(),  # type: ignore[arg-type]
    )


def _make_agent(
    *,
    resources: dict[str, list[str]] | None = None,
    subscriptions: dict[str, set[str]] | None = None,
) -> MagicMock:
    # spec=AgentProtocol makes isinstance(agent, AgentProtocol) succeed in the handler.
    agent = MagicMock(spec=AgentProtocol)
    assert isinstance(agent, AgentProtocol)
    agent.list_resources = AsyncMock(return_value=resources or {})
    agent.get_subscriptions = MagicMock(return_value=subscriptions or {})
    agent.subscribe_resource = AsyncMock()
    agent.unsubscribe_resource = AsyncMock()
    return agent


def _texts(outcome) -> str:
    return "\n".join(str(message.text) for message in outcome.messages)


@pytest.mark.asyncio
async def test_list_groups_resources_and_marks_subscribed() -> None:
    agent = _make_agent(
        resources={"demo": ["file:///a.txt", "file:///b.txt"]},
        subscriptions={"demo": {"file:///a.txt"}},
    )
    ctx = _make_context(agent)

    outcome = await handle_resources_command(
        ctx, agent_name="demo-agent", action="list", argument=None
    )

    text = _texts(outcome)
    assert "file:///a.txt" in text
    assert "[subscribed]" in text
    assert "demo" in text


@pytest.mark.asyncio
async def test_subscribe_infers_single_server() -> None:
    agent = _make_agent(resources={"demo": ["file:///a.txt"]})
    ctx = _make_context(agent)

    outcome = await handle_resources_command(
        ctx, agent_name="demo-agent", action="subscribe", argument="file:///a.txt"
    )

    agent.subscribe_resource.assert_awaited_once_with("file:///a.txt", namespace="demo")
    assert "Subscribed to" in _texts(outcome)


@pytest.mark.asyncio
async def test_subscribe_uses_explicit_server() -> None:
    agent = _make_agent(resources={"a": ["file:///x"], "b": ["file:///x"]})
    ctx = _make_context(agent)

    outcome = await handle_resources_command(
        ctx, agent_name="demo-agent", action="subscribe", argument="file:///x b"
    )

    agent.subscribe_resource.assert_awaited_once_with("file:///x", namespace="b")
    assert "server 'b'" in _texts(outcome)


@pytest.mark.asyncio
async def test_subscribe_ambiguous_server_warns_without_call() -> None:
    agent = _make_agent(resources={"a": ["file:///x"], "b": ["file:///x"]})
    ctx = _make_context(agent)

    outcome = await handle_resources_command(
        ctx, agent_name="demo-agent", action="subscribe", argument="file:///x"
    )

    agent.subscribe_resource.assert_not_awaited()
    assert "multiple servers" in _texts(outcome)


@pytest.mark.asyncio
async def test_subscribe_missing_uri_shows_usage() -> None:
    agent = _make_agent(resources={"demo": ["file:///a.txt"]})
    ctx = _make_context(agent)

    outcome = await handle_resources_command(
        ctx, agent_name="demo-agent", action="subscribe", argument=None
    )

    agent.subscribe_resource.assert_not_awaited()
    assert "Usage:" in _texts(outcome)


@pytest.mark.asyncio
async def test_subscribe_capability_error_is_surfaced() -> None:
    agent = _make_agent(resources={"demo": ["file:///a.txt"]})
    agent.subscribe_resource = AsyncMock(
        side_effect=ValueError("Server 'demo' does not support resource subscriptions")
    )
    ctx = _make_context(agent)

    outcome = await handle_resources_command(
        ctx, agent_name="demo-agent", action="subscribe", argument="file:///a.txt demo"
    )

    assert "does not support resource subscriptions" in _texts(outcome)


@pytest.mark.asyncio
async def test_subscriptions_lists_active() -> None:
    agent = _make_agent(subscriptions={"demo": {"file:///a.txt"}})
    ctx = _make_context(agent)

    outcome = await handle_resources_command(
        ctx, agent_name="demo-agent", action="subscriptions", argument=None
    )

    text = _texts(outcome)
    assert "demo" in text
    assert "file:///a.txt" in text


@pytest.mark.asyncio
async def test_unsubscribe_calls_agent() -> None:
    agent = _make_agent(resources={"demo": ["file:///a.txt"]})
    ctx = _make_context(agent)

    outcome = await handle_resources_command(
        ctx, agent_name="demo-agent", action="unsub", argument="file:///a.txt"
    )

    agent.unsubscribe_resource.assert_awaited_once_with("file:///a.txt", namespace="demo")
    assert "Unsubscribed from" in _texts(outcome)
