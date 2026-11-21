"""An MCP Events (SEP-3415) server with one event type, job.status_changed, by push or poll."""

import asyncio
from datetime import UTC, datetime
from typing import Any

from mcp.server.extension import Extension, MethodBinding
from mcp.server.mcpserver import MCPServer
from mcp.shared.exceptions import MCPError
from mcp.shared.subscriptions import SUBSCRIPTION_ID_META_KEY
from mcp_types import INVALID_PARAMS, Notification, PaginatedRequestParams
from pydantic import BaseModel

from fast_agent.mcp.events_extension import (
    EVENTS_EXTENSION,
    EVENTS_NOT_FOUND,
    EventDescriptor,
    EventOccurrence,
    EventStreamActive,
    EventStreamHeartbeat,
    ListEventsResult,
    PollEventsRequestParams,
    PollEventsResult,
    StreamEventsRequestParams,
    StreamEventsResult,
)

EVENT_NAME = "job.status_changed"
POLL_INTERVAL_MS = 1_000
HEARTBEAT_MS = 15_000

# job_id -> [(stage, timestamp)], oldest first. The cursor is an index into this list.
job_history: dict[str, list[tuple[str, str]]] = {}
# Set when any job changes stage, then replaced, so open streams wake up.
job_changed = asyncio.Event()


async def list_events(ctx: Any, params: PaginatedRequestParams) -> ListEventsResult:
    del ctx, params
    return ListEventsResult(
        events=[
            EventDescriptor(
                name=EVENT_NAME,
                description="Fires when a job changes stage",
                delivery=["push", "poll"],
                input_schema={
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}},
                    "required": ["job_id"],
                },
                payload_schema={
                    "type": "object",
                    "properties": {"job_id": {"type": "string"}, "stage": {"type": "string"}},
                },
            )
        ]
    )


def _job_id(name: str, arguments: dict[str, Any]) -> str:
    if name != EVENT_NAME:
        raise MCPError(EVENTS_NOT_FOUND, "NotFound", {"kind": "event"})
    job_id = arguments.get("job_id")
    if not isinstance(job_id, str):
        raise MCPError(INVALID_PARAMS, "job_id is required")
    return job_id


def _occurrences(job_id: str, start: int) -> list[EventOccurrence]:
    """The stage changes of a job from position ``start``, each with the cursor after it."""
    return [
        EventOccurrence(
            event_id=f"{job_id}:{index}:{stage}",
            name=EVENT_NAME,
            timestamp=timestamp,
            data={"job_id": job_id, "stage": stage},
            cursor=str(index + 1),
        )
        for index, (stage, timestamp) in enumerate(job_history.get(job_id, [])[start:], start=start)
    ]


def _start(job_id: str, cursor: str | None) -> int:
    return len(job_history.get(job_id, [])) if cursor is None else int(cursor)


async def poll_events(ctx: Any, params: PollEventsRequestParams) -> PollEventsResult:
    del ctx
    job_id = _job_id(params.name, params.arguments)
    return PollEventsResult(
        events=_occurrences(job_id, _start(job_id, params.cursor)),
        cursor=str(len(job_history.get(job_id, []))),
        next_poll_ms=POLL_INTERVAL_MS,
    )


async def stream_events(ctx: Any, params: StreamEventsRequestParams) -> StreamEventsResult:
    """Deliver a job's stage changes on the request's stream until the client cancels it."""
    job_id = _job_id(params.name, params.arguments)
    position = _start(job_id, params.cursor)
    heartbeat_ms = params.heartbeat_ms or HEARTBEAT_MS
    meta = {SUBSCRIPTION_ID_META_KEY: ctx.request_id}
    loop = asyncio.get_running_loop()

    async def send(kind: str, frame: BaseModel) -> None:
        notification = Notification[dict[str, Any], str](
            method=f"notifications/events/{kind}",
            params=frame.model_dump(by_alias=True, exclude_none=True),
        )
        await ctx.session.send_notification(notification, related_request_id=ctx.request_id)

    await send(
        "active", EventStreamActive(cursor=str(position), heartbeat_ms=heartbeat_ms, meta=meta)
    )
    quiet_until = loop.time() + heartbeat_ms / 1000
    while True:
        changed = job_changed
        for occurrence in _occurrences(job_id, position):
            await send("event", occurrence.model_copy(update={"meta": meta}))
            position += 1
            quiet_until = loop.time() + heartbeat_ms / 1000
        try:
            async with asyncio.timeout_at(quiet_until):
                await changed.wait()
        except TimeoutError:
            await send("heartbeat", EventStreamHeartbeat(cursor=str(position), meta=meta))
            quiet_until = loop.time() + heartbeat_ms / 1000


class JobEvents(Extension):
    identifier = EVENTS_EXTENSION

    def methods(self) -> list[MethodBinding]:
        return [
            MethodBinding("events/list", PaginatedRequestParams, list_events),
            MethodBinding("events/poll", PollEventsRequestParams, poll_events),
            MethodBinding("events/stream", StreamEventsRequestParams, stream_events),
        ]


server = MCPServer("events", extensions=[JobEvents()])


@server.tool()
async def set_job_stage(job_id: str, stage: str) -> str:
    """Record a new stage for a job, which fires job.status_changed."""
    global job_changed
    timestamp = datetime.now(UTC).isoformat()
    job_history.setdefault(job_id, []).append((stage, timestamp))
    job_changed.set()
    job_changed = asyncio.Event()
    return f"{job_id} is now {stage}"


if __name__ == "__main__":
    server.run(transport="stdio")
