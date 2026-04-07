from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence
from urllib.parse import urlparse, urlunparse

from fast_agent.cli.commands.url_parser import parse_server_url
from fast_agent.mcp.hf_auth import add_hf_auth_header

if TYPE_CHECKING:
    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.config import MCPServerSettings

_WILDCARD_CHARS = frozenset("*?[")
_AUTHORIZATION_HEADER_NAMES = frozenset({"authorization"})


def normalize_access_token(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if normalized.lower().startswith("bearer "):
        normalized = normalized[7:].strip()
    if not normalized:
        raise ValueError("access_token must not be empty")
    return normalized


def has_authorization_header(headers: Mapping[str, str] | None) -> bool:
    if not headers:
        return False
    return any(key.strip().lower() in _AUTHORIZATION_HEADER_NAMES for key in headers)


def normalize_client_managed_url_server(
    *,
    transport: str,
    url: str,
    headers: Mapping[str, str] | None,
    access_token: str | None,
) -> tuple[str, dict[str, str] | None]:
    final_headers = dict(headers) if headers else None

    if access_token is not None:
        if has_authorization_header(final_headers):
            raise ValueError(
                "access_token cannot be combined with headers.Authorization; "
                "use access_token or explicit Authorization headers, not both"
            )
        final_headers = dict(final_headers or {})
        final_headers["Authorization"] = f"Bearer {access_token}"

    final_url = url
    if transport == "http":
        _server_name, _transport, final_url = parse_server_url(url)

    final_headers = add_hf_auth_header(final_url, final_headers)
    return final_url, final_headers or None


def normalize_provider_managed_url_server(*, transport: str, url: str) -> str:
    final_url = url
    if transport == "http":
        _server_name, _transport, final_url = parse_server_url(url)
    return final_url


def provider_managed_base_url(url: str) -> str:
    """Return a provider-facing base URL from a normalized MCP endpoint URL."""
    parsed = urlparse(url)
    path = (parsed.path or "").rstrip("/")
    for suffix in ("/mcp", "/sse"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    clean = parsed._replace(path=path or "/", params="", query="", fragment="")
    base_url = urlunparse(clean)
    if base_url.endswith("/") and base_url.count("/") > 2:
        return base_url[:-1]
    return base_url


@dataclass(frozen=True, slots=True)
class ProviderManagedMCPAttachment:
    server_name: str
    server_description: str | None
    server_url: str
    access_token: str | None = None
    defer_loading: bool = False


@dataclass(frozen=True, slots=True)
class ProviderManagedMCPState:
    attachments: tuple[ProviderManagedMCPAttachment, ...] = ()
    tool_allowlists: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def server_names(self) -> tuple[str, ...]:
        return tuple(attachment.server_name for attachment in self.attachments)

    def has_servers(self) -> bool:
        return bool(self.attachments)


def _contains_wildcard(pattern: str) -> bool:
    return any(char in pattern for char in _WILDCARD_CHARS)


def _validate_provider_managed_server(
    *,
    server_name: str,
    settings: MCPServerSettings,
) -> None:
    invalid_fields: list[str] = []

    if not settings.url:
        invalid_fields.append("url")
    if settings.command is not None:
        invalid_fields.append("command")
    if settings.args:
        invalid_fields.append("args")
    if settings.env:
        invalid_fields.append("env")
    if settings.cwd is not None:
        invalid_fields.append("cwd")
    if settings.headers:
        invalid_fields.append("headers")
    if settings.auth is not None:
        invalid_fields.append("auth")
    if settings.roots:
        invalid_fields.append("roots")
    if settings.transport not in {"http", "sse"}:
        invalid_fields.append("transport")

    if invalid_fields:
        invalid_list = ", ".join(sorted(invalid_fields))
        raise ValueError(
            f"Provider-managed MCP server '{server_name}' has unsupported settings: {invalid_list}"
        )


def build_provider_managed_mcp_state(
    *,
    agent_config: AgentConfig,
    server_settings_by_name: Mapping[str, MCPServerSettings] | None,
) -> ProviderManagedMCPState:
    if not server_settings_by_name:
        return ProviderManagedMCPState()

    attachments: list[ProviderManagedMCPAttachment] = []
    tool_allowlists: dict[str, tuple[str, ...]] = {}
    seen_server_names: set[str] = set()

    for server_name in agent_config.servers:
        if server_name in seen_server_names:
            continue
        seen_server_names.add(server_name)

        settings = server_settings_by_name.get(server_name)
        if settings is None:
            raise ValueError(f"Unknown MCP server '{server_name}'")
        if settings.management != "provider":
            continue

        _validate_provider_managed_server(server_name=server_name, settings=settings)

        tool_patterns = tuple(agent_config.tools.get(server_name, ()))
        for tool_name in tool_patterns:
            if _contains_wildcard(tool_name):
                raise ValueError(
                    f"Provider-managed MCP server '{server_name}' requires exact tool names; "
                    f"unsupported wildcard filter: {tool_name}"
                )
        if server_name in agent_config.prompts and agent_config.prompts.get(server_name):
            raise ValueError(
                f"Provider-managed MCP server '{server_name}' does not support prompt filters"
            )
        if server_name in agent_config.resources and agent_config.resources.get(server_name):
            raise ValueError(
                f"Provider-managed MCP server '{server_name}' does not support resource filters"
            )
        if settings.url is None:
            raise ValueError(f"Provider-managed MCP server '{server_name}' requires a URL")

        if server_name in agent_config.tools:
            tool_allowlists[server_name] = tool_patterns
        attachments.append(
            ProviderManagedMCPAttachment(
                server_name=server_name,
                server_description=settings.description,
                server_url=provider_managed_base_url(settings.url),
                access_token=settings.access_token,
                defer_loading=settings.defer_loading,
            )
        )

    return ProviderManagedMCPState(
        attachments=tuple(attachments),
        tool_allowlists=tool_allowlists,
    )


def build_anthropic_provider_managed_mcp_payload(
    state: ProviderManagedMCPState,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mcp_servers: list[dict[str, Any]] = []
    toolsets: list[dict[str, Any]] = []

    for attachment in state.attachments:
        server_payload: dict[str, Any] = {
            "type": "url",
            "name": attachment.server_name,
            "url": attachment.server_url,
        }
        if attachment.access_token is not None:
            server_payload["authorization_token"] = attachment.access_token
        mcp_servers.append(server_payload)

        tool_payload: dict[str, Any] = {
            "type": "mcp_toolset",
            "mcp_server_name": attachment.server_name,
        }
        allowlist = state.tool_allowlists.get(attachment.server_name)
        if allowlist is not None:
            tool_payload["default_config"] = {"enabled": False}
            if allowlist:
                tool_payload["configs"] = {
                    tool_name: {"enabled": True} for tool_name in allowlist
                }
        toolsets.append(tool_payload)

    return mcp_servers, toolsets


def build_openai_provider_managed_mcp_tools(
    state: ProviderManagedMCPState,
) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    seen_labels: set[str] = set()

    for attachment in state.attachments:
        if attachment.server_name in seen_labels:
            raise ValueError(
                f"Duplicate provider-managed MCP server label '{attachment.server_name}'"
            )
        seen_labels.add(attachment.server_name)

        tool_payload: dict[str, Any] = {
            "type": "mcp",
            "server_label": attachment.server_name,
            "server_url": attachment.server_url,
        }
        if attachment.server_description:
            tool_payload["server_description"] = attachment.server_description
        if attachment.access_token is not None:
            tool_payload["authorization"] = attachment.access_token
        if attachment.server_name in state.tool_allowlists:
            tool_payload["allowed_tools"] = list(state.tool_allowlists[attachment.server_name])
        if attachment.defer_loading:
            tool_payload["defer_loading"] = True
        tools.append(tool_payload)

    return tools


def split_managed_server_names(
    server_names: Sequence[str],
    server_settings_by_name: Mapping[str, MCPServerSettings] | None,
) -> tuple[list[str], list[str]]:
    if not server_settings_by_name:
        return list(server_names), []

    client_managed: list[str] = []
    provider_managed: list[str] = []

    for server_name in server_names:
        settings = server_settings_by_name.get(server_name)
        if settings is not None and settings.management == "provider":
            provider_managed.append(server_name)
        else:
            client_managed.append(server_name)

    return client_managed, provider_managed
