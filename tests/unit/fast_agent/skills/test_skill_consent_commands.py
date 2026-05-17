"""Tests for `/skills pending|approve|revoke` REPL commands."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fast_agent.commands.handlers.skills import handle_skills_command
from fast_agent.skills.registry import SkillManifest


def _outcome_text(outcome) -> str:
    parts: list[str] = []
    for msg in outcome.messages:
        text = msg.text
        parts.append(text.plain if hasattr(text, "plain") else str(text))
    return "\n".join(parts)


def _ctx_with_agent(agent_obj):
    provider = MagicMock()
    provider._agent = MagicMock(return_value=agent_obj)
    ctx = SimpleNamespace(
        agent_provider=provider,
        current_agent_name="default",
        io=MagicMock(),
        settings=None,
    )
    ctx.resolve_settings = MagicMock(return_value=MagicMock())
    return ctx


def _mcp_manifest(name: str, server: str = "github") -> SkillManifest:
    return SkillManifest(
        name=name,
        description=f"The {name} skill",
        body="",
        path=None,
        uri=f"skill://{name}/SKILL.md",
        server_name=server,
    )


# --- /skills pending -----------------------------------------------------


@pytest.mark.asyncio
async def test_pending_with_nothing_pending() -> None:
    agent = SimpleNamespace(pending_skill_servers=MagicMock(return_value={}))
    ctx = _ctx_with_agent(agent)

    outcome = await handle_skills_command(
        ctx, agent_name="default", action="pending", argument=None
    )
    text = _outcome_text(outcome)
    assert "No MCP servers" in text


@pytest.mark.asyncio
async def test_pending_lists_servers_and_skills() -> None:
    pending = {
        "github": [_mcp_manifest("alpha"), _mcp_manifest("beta")],
        "acme": [_mcp_manifest("zeta", server="acme")],
    }
    agent = SimpleNamespace(pending_skill_servers=MagicMock(return_value=pending))
    ctx = _ctx_with_agent(agent)

    outcome = await handle_skills_command(
        ctx, agent_name="default", action="pending", argument=None
    )
    text = _outcome_text(outcome)
    # Server names appear.
    assert "github" in text
    assert "acme" in text
    # Skill names appear so the user can review before approving.
    assert "alpha" in text
    assert "beta" in text
    assert "zeta" in text


# --- /skills approve -----------------------------------------------------


@pytest.mark.asyncio
async def test_approve_invokes_agent() -> None:
    agent = SimpleNamespace(approve_skill_server=MagicMock(return_value=True))
    ctx = _ctx_with_agent(agent)

    outcome = await handle_skills_command(
        ctx, agent_name="default", action="approve", argument="github"
    )
    text = _outcome_text(outcome)
    assert "Approved" in text
    agent.approve_skill_server.assert_called_once_with("github")


@pytest.mark.asyncio
async def test_approve_requires_argument() -> None:
    agent = SimpleNamespace(approve_skill_server=MagicMock(return_value=True))
    ctx = _ctx_with_agent(agent)

    outcome = await handle_skills_command(
        ctx, agent_name="default", action="approve", argument=None
    )
    text = _outcome_text(outcome)
    assert "Usage:" in text
    agent.approve_skill_server.assert_not_called()


@pytest.mark.asyncio
async def test_approve_unknown_server_warns() -> None:
    """A typo or a server with nothing pending must produce a clear
    warning, not a silent success that misleads the user into thinking
    consent was recorded."""
    agent = SimpleNamespace(approve_skill_server=MagicMock(return_value=False))
    ctx = _ctx_with_agent(agent)

    outcome = await handle_skills_command(
        ctx, agent_name="default", action="approve", argument="nope"
    )
    text = _outcome_text(outcome)
    assert "No skill catalog" in text or "pending" in text.lower()


# --- /skills revoke ------------------------------------------------------


@pytest.mark.asyncio
async def test_revoke_invokes_agent() -> None:
    agent = SimpleNamespace(revoke_skill_server=MagicMock(return_value=True))
    ctx = _ctx_with_agent(agent)

    outcome = await handle_skills_command(
        ctx, agent_name="default", action="revoke", argument="github"
    )
    text = _outcome_text(outcome)
    assert "Revoked" in text
    agent.revoke_skill_server.assert_called_once_with("github")


@pytest.mark.asyncio
async def test_revoke_requires_argument() -> None:
    agent = SimpleNamespace(revoke_skill_server=MagicMock(return_value=True))
    ctx = _ctx_with_agent(agent)

    outcome = await handle_skills_command(
        ctx, agent_name="default", action="revoke", argument=None
    )
    text = _outcome_text(outcome)
    assert "Usage:" in text


@pytest.mark.asyncio
async def test_revoke_unknown_server_warns() -> None:
    agent = SimpleNamespace(revoke_skill_server=MagicMock(return_value=False))
    ctx = _ctx_with_agent(agent)

    outcome = await handle_skills_command(
        ctx, agent_name="default", action="revoke", argument="nope"
    )
    text = _outcome_text(outcome)
    assert "No active or pending consent" in text or "nope" in text


# --- unknown action ------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_action_lists_new_verbs() -> None:
    """The fallback error message must mention the new approve/revoke/
    pending verbs so users can discover them via typos."""
    ctx = _ctx_with_agent(SimpleNamespace())
    outcome = await handle_skills_command(
        ctx, agent_name="default", action="bogus", argument=None
    )
    text = _outcome_text(outcome)
    assert "pending" in text
    assert "approve" in text
    assert "revoke" in text
