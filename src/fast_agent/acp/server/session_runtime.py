from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, Sequence

from acp.helpers import update_agent_message_text
from acp.schema import HttpMcpServer, McpServerStdio, SessionMode, SessionModeState, SseMcpServer

from fast_agent.acp.acp_context import ACPContext, ClientInfo
from fast_agent.acp.acp_context import ClientCapabilities as FAClientCapabilities
from fast_agent.acp.filesystem_runtime import ACPFilesystemRuntime
from fast_agent.acp.permission_store import PermissionStore
from fast_agent.acp.protocols import (
    FilesystemRuntimeCapable,
    InstructionContextCapable,
    PlanTelemetryCapable,
    ShellRuntimeCapable,
    WorkflowTelemetryCapable,
)
from fast_agent.acp.server.common import (
    coerce_registry_version,
    format_agent_name_as_title,
    truncate_description,
)
from fast_agent.acp.server.models import ACPSessionState
from fast_agent.acp.terminal_runtime import ACPTerminalRuntime
from fast_agent.acp.tool_permission_adapter import ACPToolPermissionAdapter
from fast_agent.acp.tool_progress import ACPToolProgressManager
from fast_agent.agents.tool_runner import ToolRunnerHooks
from fast_agent.context import Context
from fast_agent.core.instruction_refresh import (
    McpInstructionCapable,
    build_instruction,
    resolve_instruction_skill_manifests,
)
from fast_agent.core.instruction_utils import (
    build_agent_instruction_context,
    get_instruction_template,
)
from fast_agent.core.logging.logger import get_logger
from fast_agent.core.prompt_templates import enrich_with_environment_context
from fast_agent.interfaces import ACPAwareProtocol
from fast_agent.llm.usage_tracking import last_turn_usage
from fast_agent.mcp.tool_execution_handler import NoOpToolExecutionHandler
from fast_agent.mcp.tool_permission_handler import NoOpToolPermissionHandler
from fast_agent.mcp.types import McpAgentProtocol
from fast_agent.types import RequestParams
from fast_agent.workflow_telemetry import ACPPlanTelemetryProvider, ToolHandlerWorkflowTelemetry

if TYPE_CHECKING:
    from fast_agent.core.fastagent import AgentInstance

logger = get_logger(__name__)


class SessionRuntimeHost(Protocol):
    primary_instance: AgentInstance
    _create_instance_task: Any
    _dispose_instance_task: Any
    _instance_scope: str
    _get_registry_version: Any
    _primary_registry_version: int
    _shared_reload_lock: asyncio.Lock
    _stale_instances: list[AgentInstance]
    server_name: str
    sessions: dict[str, AgentInstance]
    _session_lock: asyncio.Lock
    _prompt_locks: dict[str, asyncio.Lock]
    _session_state: dict[str, ACPSessionState]
    _connection: Any
    _client_supports_terminal: bool
    _client_supports_fs_read: bool
    _client_supports_fs_write: bool
    _client_info: dict[str, Any] | None
    _skills_directory_override: Sequence[str] | str | None
    _parsed_client_capabilities: FAClientCapabilities | None
    _parsed_client_info: ClientInfo | None
    _protocol_version: int | None
    primary_agent_name: str | None
    _permissions_enabled: bool

    def _calculate_terminal_output_limit(self, agent: Any) -> int: ...

    def _select_primary_agent(self, instance: AgentInstance) -> str | None: ...

    def _create_slash_handler(
        self,
        session_state: ACPSessionState,
        instance: AgentInstance,
    ) -> Any: ...

    async def _send_available_commands_update(self, session_id: str) -> None: ...


class ACPServerSessionRuntime:
    def __init__(self, host: SessionRuntimeHost) -> None:
        self._host = host

    def build_session_modes(
        self, instance: AgentInstance, session_state: ACPSessionState | None = None
    ) -> SessionModeState:
        available_modes: list[SessionMode] = []
        resolved_cache = session_state.resolved_instructions if session_state else {}
        tool_only_agents = getattr(instance.app, "_tool_only_agents", set())

        for agent_name, agent in instance.agents.items():
            if agent_name in tool_only_agents:
                continue

            instruction = resolved_cache.get(agent_name) or agent.instruction
            description = truncate_description(instruction) if instruction else None
            display_name = format_agent_name_as_title(agent_name)

            if isinstance(agent, ACPAwareProtocol):
                try:
                    mode_info = agent.acp_mode_info()
                except Exception:
                    logger.warning(
                        "Error getting acp_mode_info from agent",
                        name="acp_mode_info_error",
                        agent_name=agent_name,
                        exc_info=True,
                    )
                    mode_info = None

                if mode_info:
                    if mode_info.name:
                        display_name = mode_info.name
                    if mode_info.description:
                        description = mode_info.description

            if description:
                description = truncate_description(description)

            available_modes.append(
                SessionMode(
                    id=agent_name,
                    name=display_name,
                    description=description,
                )
            )

        current_mode_id = self._host.primary_agent_name or (
            list(instance.agents.keys())[0] if instance.agents else "default"
        )
        return SessionModeState(
            available_modes=available_modes,
            current_mode_id=current_mode_id,
        )

    async def build_session_request_params(
        self, agent: Any, session_state: ACPSessionState | None
    ) -> RequestParams | None:
        if not getattr(agent, "_llm", None):
            return None

        resolved_cache = session_state.resolved_instructions if session_state else {}
        resolved = resolved_cache.get(getattr(agent, "name", ""), None)
        if isinstance(agent, McpInstructionCapable) or resolved is None:
            context = session_state.prompt_context if session_state else None
            if not context:
                return None
            resolved = await self.resolve_instruction_for_session(agent, context)
            if not resolved:
                return None
            if session_state is not None:
                session_state.resolved_instructions[getattr(agent, "name", "")] = resolved
        return RequestParams(systemPrompt=resolved)

    async def resolve_instruction_for_session(
        self,
        agent: object,
        context: dict[str, str],
    ) -> str | None:
        template = get_instruction_template(agent)
        if not template:
            return None

        aggregator = None
        skill_manifests = None
        skill_read_tool_name = "read_skill"
        effective_context = dict(context)
        if isinstance(agent, McpInstructionCapable):
            aggregator = agent.aggregator
            skill_manifests = resolve_instruction_skill_manifests(agent, agent.skill_manifests)
            skill_read_tool_name = agent.skill_read_tool_name
            if agent.instruction_context:
                effective_context = dict(agent.instruction_context)

        effective_context = build_agent_instruction_context(agent, effective_context)
        return await build_instruction(
            template,
            aggregator=aggregator,
            skill_manifests=skill_manifests,
            skill_read_tool_name=skill_read_tool_name,
            context=effective_context,
            source=getattr(agent, "name", None),
        )

    async def maybe_refresh_shared_instance(self) -> None:
        if self._host._instance_scope != "shared" or not self._host._get_registry_version:
            return
        if self._host._session_state and any(
            session_id for session_id in self._host._session_state if session_id
        ):
            # Prompts guard refresh with _active_prompts; keep the original fast path.
            pass
        if getattr(self._host, "_active_prompts", None):
            return

        latest_version = coerce_registry_version(self._host._get_registry_version())
        if latest_version <= self._host._primary_registry_version:
            return

        async with self._host._shared_reload_lock:
            if getattr(self._host, "_active_prompts", None):
                return
            latest_version = coerce_registry_version(self._host._get_registry_version())
            if latest_version <= self._host._primary_registry_version:
                return

            new_instance = await self._host._create_instance_task()
            old_instance = self._host.primary_instance
            self._host.primary_instance = new_instance
            self._host._primary_registry_version = coerce_registry_version(
                getattr(new_instance, "registry_version", latest_version)
            )
            self._host._stale_instances.append(old_instance)
            self._host.primary_agent_name = self._host._select_primary_agent(new_instance)
            await self.refresh_sessions_for_instance(new_instance)

    async def replace_instance_for_session(
        self,
        session_state: ACPSessionState,
        *,
        dispose_error_name: str,
        await_refresh_session_state: bool,
    ) -> AgentInstance:
        if self._host._instance_scope == "shared":
            async with self._host._shared_reload_lock:
                new_instance = await self._host._create_instance_task()
                old_instance = self._host.primary_instance
                self._host.primary_instance = new_instance
                latest_version = (
                    coerce_registry_version(self._host._get_registry_version())
                    if self._host._get_registry_version
                    else 0
                )
                self._host._primary_registry_version = coerce_registry_version(
                    getattr(new_instance, "registry_version", latest_version)
                )
                self._host._stale_instances.append(old_instance)
                self._host.primary_agent_name = self._host._select_primary_agent(new_instance)
                await self.refresh_sessions_for_instance(new_instance)
            return session_state.instance

        instance = await self._host._create_instance_task()
        old_instance = session_state.instance
        session_state.instance = instance
        async with self._host._session_lock:
            self._host.sessions[session_state.session_id] = instance
        if await_refresh_session_state:
            await self.refresh_session_state(session_state, instance)
        else:
            asyncio.create_task(self.refresh_session_state(session_state, instance))
        if old_instance != self._host.primary_instance:
            try:
                await self._host._dispose_instance_task(old_instance)
            except Exception as exc:
                logger.warning(
                    "Failed to dispose old session instance",
                    name=dispose_error_name,
                    session_id=session_state.session_id,
                    error=str(exc),
                )
        return instance

    async def refresh_sessions_for_instance(self, instance: AgentInstance) -> None:
        async with self._host._session_lock:
            for session_id, session_state in self._host._session_state.items():
                self._host.sessions[session_id] = instance
                session_state.instance = instance
                await self.refresh_session_state(session_state, instance)

    async def refresh_session_state(
        self, session_state: ACPSessionState, instance: AgentInstance
    ) -> None:
        prompt_context = session_state.prompt_context or {}
        resolved_for_session: dict[str, str] = {}
        for agent_name, agent in instance.agents.items():
            resolved = await self.resolve_instruction_for_session(agent, prompt_context)
            if resolved:
                resolved_for_session[agent_name] = resolved
        session_state.resolved_instructions = resolved_for_session

        for agent_name, agent in instance.agents.items():
            if isinstance(agent, InstructionContextCapable):
                try:
                    context_with_agent = build_agent_instruction_context(agent, prompt_context)
                    agent.set_instruction_context(context_with_agent)
                except Exception as exc:
                    logger.warning(
                        "Failed to set instruction context on agent",
                        name="acp_instruction_context_failed",
                        session_id=session_state.session_id,
                        agent_name=agent_name,
                        error=str(exc),
                    )

        tool_handler = session_state.progress_manager
        permission_handler = session_state.permission_handler
        if tool_handler:
            workflow_telemetry = ToolHandlerWorkflowTelemetry(
                tool_handler, server_name=self._host.server_name
            )
            for agent_name, agent in instance.agents.items():
                if isinstance(agent, McpAgentProtocol):
                    aggregator = agent.aggregator
                    if isinstance(aggregator._tool_handler, NoOpToolExecutionHandler):
                        aggregator._tool_handler = tool_handler
                        logger.debug(
                            "ACP tool handler registered (refresh)",
                            name="acp_tool_handler_refresh",
                            session_id=session_state.session_id,
                            agent_name=agent_name,
                        )

                if isinstance(agent, WorkflowTelemetryCapable) and agent.workflow_telemetry is None:
                    agent.workflow_telemetry = workflow_telemetry

                if isinstance(agent, PlanTelemetryCapable):
                    if agent.plan_telemetry is None and self._host._connection:
                        plan_telemetry = ACPPlanTelemetryProvider(
                            self._host._connection, session_state.session_id
                        )
                        agent.plan_telemetry = plan_telemetry

                llm = getattr(agent, "_llm", None)
                if llm and hasattr(llm, "add_tool_stream_listener"):
                    try:
                        llm.add_tool_stream_listener(tool_handler.handle_tool_stream_event)
                    except Exception:
                        pass

        if permission_handler:
            for agent_name, agent in instance.agents.items():
                if isinstance(agent, McpAgentProtocol):
                    aggregator = agent.aggregator
                    if isinstance(aggregator._permission_handler, NoOpToolPermissionHandler):
                        aggregator._permission_handler = permission_handler
                        logger.debug(
                            "ACP permission handler registered (refresh)",
                            name="acp_permission_handler_refresh",
                            session_id=session_state.session_id,
                            agent_name=agent_name,
                        )

        if session_state.terminal_runtime:
            for agent in instance.agents.values():
                if isinstance(agent, ShellRuntimeCapable) and agent._shell_runtime_enabled:
                    agent.set_external_runtime(session_state.terminal_runtime)

        if session_state.filesystem_runtime:
            for agent in instance.agents.values():
                if isinstance(agent, FilesystemRuntimeCapable):
                    agent.set_filesystem_runtime(session_state.filesystem_runtime)
            from fast_agent.core.instruction_refresh import rebuild_agent_instruction

            for agent in instance.agents.values():
                await rebuild_agent_instruction(agent)

        slash_handler = self._host._create_slash_handler(session_state, instance)
        session_state.slash_handler = slash_handler

        current_agent = session_state.current_agent_name
        if not current_agent or current_agent not in instance.agents:
            current_agent = self._host.primary_agent_name or next(iter(instance.agents.keys()), None)
            session_state.current_agent_name = current_agent
        if current_agent and session_state.slash_handler:
            session_state.slash_handler.set_current_agent(current_agent)

        session_modes = self.build_session_modes(instance, session_state)
        if current_agent and current_agent in instance.agents:
            session_modes = SessionModeState(
                available_modes=session_modes.available_modes,
                current_mode_id=current_agent,
            )

        if session_state.acp_context:
            slash_handler.set_acp_context(session_state.acp_context)
            session_state.acp_context.set_slash_handler(slash_handler)
            session_state.acp_context.set_resolved_instructions(resolved_for_session)
            session_state.acp_context.set_available_modes(session_modes.available_modes)
            if current_agent:
                session_state.acp_context.set_current_mode(current_agent)

    async def initialize_session_state(
        self,
        session_id: str,
        *,
        cwd: str,
        mcp_servers: list[HttpMcpServer | SseMcpServer | McpServerStdio],
    ) -> tuple[ACPSessionState, SessionModeState]:
        _ = mcp_servers
        await self.maybe_refresh_shared_instance()

        async with self._host._session_lock:
            session_state = self._host._session_state.get(session_id)
            if session_state:
                instance = session_state.instance
                self._host.sessions[session_id] = instance
            else:
                if self._host._instance_scope == "shared":
                    instance = self._host.primary_instance
                elif self._host._instance_scope in ["connection", "request"]:
                    instance = await self._host._create_instance_task()
                else:
                    instance = self._host.primary_instance

                self._host.sessions[session_id] = instance
                session_state = ACPSessionState(session_id=session_id, instance=instance)
                self._host._session_state[session_id] = session_state

            session_state.session_cwd = cwd
            if session_state.session_store_scope == "workspace":
                session_state.session_store_cwd = cwd

            if session_id not in self._host._prompt_locks:
                self._host._prompt_locks[session_id] = asyncio.Lock()

            tool_handler = session_state.progress_manager
            if self._host._connection and tool_handler is None:
                tool_handler = ACPToolProgressManager(self._host._connection, session_id)
                session_state.progress_manager = tool_handler
                workflow_telemetry = ToolHandlerWorkflowTelemetry(
                    tool_handler, server_name=self._host.server_name
                )
                logger.info(
                    "ACP tool progress manager created for session",
                    name="acp_tool_progress_init",
                    session_id=session_id,
                )

                for agent_name, agent in instance.agents.items():
                    if isinstance(agent, McpAgentProtocol):
                        agent.aggregator._tool_handler = tool_handler
                        logger.info(
                            "ACP tool handler registered",
                            name="acp_tool_handler_registered",
                            session_id=session_id,
                            agent_name=agent_name,
                        )

                    if isinstance(agent, WorkflowTelemetryCapable):
                        agent.workflow_telemetry = workflow_telemetry

                    if isinstance(agent, PlanTelemetryCapable):
                        plan_telemetry = ACPPlanTelemetryProvider(
                            self._host._connection, session_id
                        )
                        agent.plan_telemetry = plan_telemetry
                        logger.info(
                            "ACP plan telemetry registered",
                            name="acp_plan_telemetry_registered",
                            session_id=session_id,
                            agent_name=agent_name,
                        )

                    llm = getattr(agent, "_llm", None)
                    if llm and hasattr(llm, "add_tool_stream_listener"):
                        try:
                            llm.add_tool_stream_listener(tool_handler.handle_tool_stream_event)
                            logger.info(
                                "ACP tool handler registered as stream listener",
                                name="acp_tool_stream_listener_registered",
                                session_id=session_id,
                                agent_name=agent_name,
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to register tool stream listener: {e}",
                                name="acp_tool_stream_listener_failed",
                                exc_info=True,
                            )

            if (
                self._host._connection
                and self._host._permissions_enabled
                and session_state.permission_handler is None
            ):
                session_cwd = cwd or "."
                permission_store = PermissionStore(cwd=session_cwd)
                permission_handler = ACPToolPermissionAdapter(
                    connection=self._host._connection,
                    session_id=session_id,
                    store=permission_store,
                    cwd=session_cwd,
                    tool_handler=tool_handler,
                )
                session_state.permission_handler = permission_handler

                for agent_name, agent in instance.agents.items():
                    if isinstance(agent, McpAgentProtocol):
                        agent.aggregator._permission_handler = permission_handler
                        logger.info(
                            "ACP permission handler registered",
                            name="acp_permission_handler_registered",
                            session_id=session_id,
                            agent_name=agent_name,
                        )

                logger.info(
                    "ACP tool permissions enabled for session",
                    name="acp_permissions_init",
                    session_id=session_id,
                    cwd=cwd,
                )

            if (
                self._host._connection
                and self._host._client_supports_terminal
                and session_state.terminal_runtime is None
            ):
                for agent_name, agent in instance.agents.items():
                    if isinstance(agent, ShellRuntimeCapable) and agent._shell_runtime_enabled:
                        default_limit = getattr(
                            agent._shell_runtime,
                            "_output_byte_limit",
                            self._host._calculate_terminal_output_limit(agent),
                        )
                        perm_handler = session_state.permission_handler
                        terminal_runtime = ACPTerminalRuntime(
                            connection=self._host._connection,
                            session_id=session_id,
                            activation_reason="via ACP terminal support",
                            timeout_seconds=getattr(agent._shell_runtime, "timeout_seconds", 90),
                            tool_handler=tool_handler,
                            default_output_byte_limit=default_limit,
                            permission_handler=perm_handler,
                        )
                        agent.set_external_runtime(terminal_runtime)
                        session_state.terminal_runtime = terminal_runtime

                        logger.info(
                            "ACP terminal runtime injected",
                            name="acp_terminal_injected",
                            session_id=session_id,
                            agent_name=agent_name,
                            default_output_limit=default_limit,
                        )

            if (
                self._host._connection
                and (self._host._client_supports_fs_read or self._host._client_supports_fs_write)
                and session_state.filesystem_runtime is None
            ):
                perm_handler = session_state.permission_handler
                filesystem_runtime = ACPFilesystemRuntime(
                    connection=self._host._connection,
                    session_id=session_id,
                    activation_reason="via ACP filesystem support",
                    enable_read=self._host._client_supports_fs_read,
                    enable_write=self._host._client_supports_fs_write,
                    tool_handler=tool_handler,
                    permission_handler=perm_handler,
                )
                session_state.filesystem_runtime = filesystem_runtime

                for agent_name, agent in instance.agents.items():
                    if isinstance(agent, FilesystemRuntimeCapable):
                        agent.set_filesystem_runtime(filesystem_runtime)
                        logger.info(
                            "ACP filesystem runtime injected",
                            name="acp_filesystem_injected",
                            session_id=session_id,
                            agent_name=agent_name,
                            read_enabled=self._host._client_supports_fs_read,
                            write_enabled=self._host._client_supports_fs_write,
                        )

                from fast_agent.core.instruction_refresh import rebuild_agent_instruction

                for agent in instance.agents.values():
                    await rebuild_agent_instruction(agent)

        session_context: dict[str, str] = {}
        enrich_with_environment_context(
            session_context,
            cwd,
            self._host._client_info,
            self._host._skills_directory_override,
        )
        session_state.prompt_context = session_context

        resolved_for_session: dict[str, str] = {}
        for agent_name, agent in instance.agents.items():
            resolved = await self.resolve_instruction_for_session(agent, session_context)
            if resolved:
                resolved_for_session[agent_name] = resolved
        if resolved_for_session:
            session_state.resolved_instructions = resolved_for_session

        for agent_name, agent in instance.agents.items():
            if isinstance(agent, InstructionContextCapable):
                try:
                    context_with_agent = build_agent_instruction_context(agent, session_context)
                    agent.set_instruction_context(context_with_agent)
                except Exception as e:
                    logger.warning(f"Failed to set instruction context on agent {agent_name}: {e}")

        slash_handler = self._host._create_slash_handler(session_state, instance)
        session_state.slash_handler = slash_handler

        acp_context = session_state.acp_context
        if self._host._connection:
            if acp_context is None:
                acp_context = ACPContext(
                    connection=self._host._connection,
                    session_id=session_id,
                    session_cwd=cwd,
                    session_store_scope=session_state.session_store_scope,
                    session_store_cwd=session_state.session_store_cwd,
                    client_capabilities=self._host._parsed_client_capabilities,
                    client_info=self._host._parsed_client_info,
                    protocol_version=self._host._protocol_version,
                )
                session_state.acp_context = acp_context
            else:
                acp_context.set_session_cwd(cwd)
                acp_context.set_session_store(
                    session_state.session_store_scope,
                    session_state.session_store_cwd,
                )

            if session_state.terminal_runtime:
                acp_context.set_terminal_runtime(session_state.terminal_runtime)
            if session_state.filesystem_runtime:
                acp_context.set_filesystem_runtime(session_state.filesystem_runtime)
            if session_state.permission_handler:
                acp_context.set_permission_handler(session_state.permission_handler)
            if session_state.progress_manager:
                acp_context.set_progress_manager(session_state.progress_manager)

            slash_handler.set_acp_context(acp_context)
            acp_context.set_slash_handler(slash_handler)
            acp_context.set_resolved_instructions(session_state.resolved_instructions)

            for agent_name, agent in instance.agents.items():
                context = getattr(agent, "context", None)
                if isinstance(context, Context):
                    context.acp = acp_context
                    logger.debug(
                        "ACPContext set on agent",
                        name="acp_context_set",
                        session_id=session_id,
                        agent_name=agent_name,
                    )

            logger.info(
                "ACPContext created for session",
                name="acp_context_created",
                session_id=session_id,
                has_terminal=acp_context.terminal_runtime is not None,
                has_filesystem=acp_context.filesystem_runtime is not None,
                has_permissions=acp_context.permission_handler is not None,
            )

        if self._host._connection:
            asyncio.create_task(self._host._send_available_commands_update(session_id))

        session_modes = self.build_session_modes(instance, session_state)
        session_state.current_agent_name = session_modes.currentModeId
        if session_state.acp_context:
            session_state.acp_context.set_available_modes(session_modes.availableModes)
            session_state.acp_context.set_current_mode(session_modes.currentModeId)

        logger.info(
            "Session modes initialized",
            name="acp_session_modes_init",
            session_id=session_id,
            current_mode=session_modes.currentModeId,
            mode_count=len(session_modes.availableModes),
        )

        return session_state, session_modes

    async def dispose_stale_instances_if_idle(self) -> None:
        if getattr(self._host, "_active_prompts", None):
            return
        if not self._host._stale_instances:
            return
        stale = list(self._host._stale_instances)
        self._host._stale_instances.clear()
        for instance in stale:
            await self._host._dispose_instance_task(instance)

    def build_status_line_meta(
        self, agent: Any, turn_start_index: int | None
    ) -> dict[str, Any] | None:
        if not agent or not getattr(agent, "usage_accumulator", None):
            return None

        totals = last_turn_usage(agent.usage_accumulator, turn_start_index)
        if not totals:
            return None

        input_tokens = totals["input_tokens"]
        output_tokens = totals["output_tokens"]
        tool_calls = totals["tool_calls"]
        tool_info = f", {tool_calls} tools" if tool_calls > 0 else ""
        context_pct = agent.usage_accumulator.context_usage_percentage
        context_info = f" ({context_pct:.1f}%)" if context_pct is not None else ""
        status_line = f"{input_tokens:,} in, {output_tokens:,} out{tool_info}{context_info}"
        return {"field_meta": {"openhands.dev/metrics": {"status_line": status_line}}}

    @staticmethod
    def merge_tool_runner_hooks(
        base: ToolRunnerHooks | None, extra: ToolRunnerHooks | None
    ) -> ToolRunnerHooks | None:
        if base is None:
            return extra
        if extra is None:
            return base

        def merge(one: Any, two: Any) -> Any:
            if one is None:
                return two
            if two is None:
                return one

            async def merged(*args: Any, **kwargs: Any) -> None:
                await one(*args, **kwargs)
                await two(*args, **kwargs)

            return merged

        return ToolRunnerHooks(
            before_llm_call=merge(base.before_llm_call, extra.before_llm_call),
            after_llm_call=merge(base.after_llm_call, extra.after_llm_call),
            before_tool_call=merge(base.before_tool_call, extra.before_tool_call),
            after_tool_call=merge(base.after_tool_call, extra.after_tool_call),
            after_turn_complete=merge(base.after_turn_complete, extra.after_turn_complete),
        )

    async def send_status_line_update(
        self, session_id: str, agent: Any, turn_start_index: int | None
    ) -> None:
        if not self._host._connection:
            return
        status_line_meta = self.build_status_line_meta(agent, turn_start_index)
        if not status_line_meta:
            return
        try:
            message_chunk = update_agent_message_text("")
            await self._host._connection.session_update(
                session_id=session_id,
                update=message_chunk,
                **status_line_meta,
            )
        except Exception as exc:
            logger.error(
                f"Error sending status line update: {exc}",
                name="acp_status_line_update_error",
                exc_info=True,
            )
