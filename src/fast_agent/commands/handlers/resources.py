"""Shared resources command handlers (list / subscribe / unsubscribe)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text

from fast_agent.commands.results import CommandOutcome
from fast_agent.interfaces import AgentProtocol

if TYPE_CHECKING:
    from fast_agent.commands.context import CommandContext


_RESOURCES_RIGHT_INFO = "resources"

_USAGE_LINES = (
    "/resources                       List resources (grouped by server)",
    "/resources subscribe <uri> [server]    Subscribe to resource updates",
    "/resources unsubscribe <uri> [server]  Cancel a resource subscription",
    "/resources subscriptions         Show active subscriptions",
    "/resources help                  Show this help",
)

_ACTION_ALIASES = {
    "ls": "list",
    "sub": "subscribe",
    "unsub": "unsubscribe",
    "subs": "subscriptions",
    "--help": "help",
    "-h": "help",
}


def _warn(outcome: CommandOutcome, message: str, agent_name: str) -> CommandOutcome:
    outcome.add_message(
        message,
        channel="warning",
        right_info=_RESOURCES_RIGHT_INFO,
        agent_name=agent_name,
    )
    return outcome


def _usage(outcome: CommandOutcome, agent_name: str) -> CommandOutcome:
    content = Text()
    content.append("Resource subscriptions\n\n", style="bold")
    for line in _USAGE_LINES:
        content.append(line + "\n", style="white")
    outcome.add_message(content, right_info=_RESOURCES_RIGHT_INFO, agent_name=agent_name)
    return outcome


def _split_argument(argument: str | None) -> tuple[str | None, str | None]:
    """Split a command argument into (resource_uri, server_name)."""
    if not argument:
        return None, None
    tokens = argument.split()
    uri = tokens[0] if tokens else None
    server = tokens[1] if len(tokens) > 1 else None
    return uri, server


async def _resolve_server_for_uri(
    agent: AgentProtocol, resource_uri: str
) -> tuple[str | None, list[str]]:
    """
    Find which server exposes a resource URI.

    Returns (server_name, candidate_servers). server_name is set only when exactly one
    server lists the URI; candidate_servers lists every server that does.
    """
    resources = await agent.list_resources()
    candidates = [server for server, uris in resources.items() if resource_uri in uris]
    if len(candidates) == 1:
        return candidates[0], candidates
    # Fall back to the only configured server when the URI isn't enumerable.
    if not candidates and len(resources) == 1:
        only_server = next(iter(resources))
        return only_server, [only_server]
    return None, candidates


async def handle_resources_command(
    ctx: "CommandContext",
    *,
    agent_name: str,
    action: str,
    argument: str | None,
) -> CommandOutcome:
    outcome = CommandOutcome()

    agent = ctx.agent_provider._agent(agent_name)
    if not isinstance(agent, AgentProtocol):
        return _warn(outcome, "This agent does not support resources.", agent_name)

    normalized = _ACTION_ALIASES.get(action.lower(), action.lower())

    if normalized == "help":
        return _usage(outcome, agent_name)

    if normalized == "list":
        return await _handle_list(outcome, agent, agent_name)

    if normalized == "subscriptions":
        return _handle_subscriptions(outcome, agent, agent_name)

    if normalized in {"subscribe", "unsubscribe"}:
        return await _handle_subscribe(
            outcome, agent, agent_name, argument, subscribe=(normalized == "subscribe")
        )

    return _usage(_warn(outcome, f"Unknown resources action: {action}", agent_name), agent_name)


async def _handle_list(
    outcome: CommandOutcome, agent: AgentProtocol, agent_name: str
) -> CommandOutcome:
    resources = await agent.list_resources()
    subscriptions = agent.get_subscriptions()

    if not resources:
        return _warn(outcome, "No resources available for this agent.", agent_name)

    content = Text()
    content.append(f"Resources for agent {agent_name}:\n\n", style="bold")
    for server in sorted(resources):
        uris = resources[server]
        subscribed = subscriptions.get(server, set())
        content.append(f"{server}", style="bright_blue bold")
        content.append(f" ({len(uris)})\n", style="dim cyan")
        if not uris:
            content.append("  (none)\n", style="dim")
        for uri in uris:
            marker = " [subscribed]" if uri in subscribed else ""
            content.append("  • ", style="dim cyan")
            content.append(uri, style="white")
            if marker:
                content.append(marker, style="green")
            content.append("\n")
        content.append("\n")

    outcome.add_message(content, right_info=_RESOURCES_RIGHT_INFO, agent_name=agent_name)
    return outcome


def _handle_subscriptions(
    outcome: CommandOutcome, agent: AgentProtocol, agent_name: str
) -> CommandOutcome:
    subscriptions = agent.get_subscriptions()
    active = {server: uris for server, uris in subscriptions.items() if uris}

    if not active:
        return _warn(outcome, "No active resource subscriptions.", agent_name)

    content = Text()
    content.append("Active resource subscriptions:\n\n", style="bold")
    for server in sorted(active):
        content.append(f"{server}\n", style="bright_blue bold")
        for uri in sorted(active[server]):
            content.append("  • ", style="dim cyan")
            content.append(uri, style="white")
            content.append("\n")
        content.append("\n")

    outcome.add_message(content, right_info=_RESOURCES_RIGHT_INFO, agent_name=agent_name)
    return outcome


async def _handle_subscribe(
    outcome: CommandOutcome,
    agent: AgentProtocol,
    agent_name: str,
    argument: str | None,
    *,
    subscribe: bool,
) -> CommandOutcome:
    verb = "subscribe" if subscribe else "unsubscribe"
    resource_uri, server_name = _split_argument(argument)

    if not resource_uri:
        return _warn(
            outcome,
            f"Usage: /resources {verb} <uri> [server]",
            agent_name,
        )

    if server_name is None:
        server_name, candidates = await _resolve_server_for_uri(agent, resource_uri)
        if server_name is None:
            if candidates:
                joined = ", ".join(sorted(candidates))
                return _warn(
                    outcome,
                    f"Resource '{resource_uri}' is available on multiple servers ({joined}). "
                    f"Specify one: /resources {verb} {resource_uri} <server>",
                    agent_name,
                )
            return _warn(
                outcome,
                f"Could not determine which server owns '{resource_uri}'. "
                f"Specify one: /resources {verb} {resource_uri} <server>",
                agent_name,
            )

    delivers_updates = True
    try:
        if subscribe:
            delivers_updates = await agent.subscribe_resource(resource_uri, namespace=server_name)
        else:
            await agent.unsubscribe_resource(resource_uri, namespace=server_name)
    except ValueError as exc:
        return _warn(outcome, str(exc), agent_name)

    past = "Subscribed to" if subscribe else "Unsubscribed from"
    outcome.add_message(
        f"{past} '{resource_uri}' on server '{server_name}'.",
        channel="info",
        right_info=_RESOURCES_RIGHT_INFO,
        agent_name=agent_name,
    )
    if subscribe and not delivers_updates:
        _warn(
            outcome,
            "Note: this server is connected without persistent connections, so resource "
            "update notifications will not be delivered. Re-read the resource to get changes.",
            agent_name,
        )
    return outcome
