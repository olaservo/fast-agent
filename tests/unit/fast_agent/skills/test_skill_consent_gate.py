"""Integration tests for the per-server skill-consent gate on McpAgent.

These tests exercise `_apply_consent_gate`, `approve_skill_server`,
`revoke_skill_server`, and `pending_skill_servers` together — the gate
is the actual feature; the consent store is its persistence layer.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.mcp_agent import McpAgent
from fast_agent.context import Context
from fast_agent.skills.consent import (
    SkillConsentStore,
    compute_catalog_fingerprint,
    default_consent_path,
)
from fast_agent.skills.registry import SkillManifest


def _mcp_manifest(name: str, server: str = "github") -> SkillManifest:
    return SkillManifest(
        name=name,
        description=f"The {name} skill",
        body="",
        path=None,
        uri=f"skill://{name}/SKILL.md",
        server_name=server,
    )


def _agent_with_home(tmp_path: Path) -> McpAgent:
    """Build an agent whose consent store writes under `tmp_path`.

    The `_fast_agent_home` private attr drives `_consent_store()` —
    setting it points the store at a hermetic location.
    """
    config = AgentConfig(name="test", instruction="x", servers=[], skills=tmp_path)
    config.skill_manifests = []
    ctx = Context()
    if ctx.config is None:
        # Construct a minimal settings-like object so _consent_store can read the home.
        from fast_agent.config import Settings

        ctx.config = Settings()
    ctx.config._fast_agent_home = str(tmp_path)
    return McpAgent(config=config, context=ctx)


# --- gate partitioning ---------------------------------------------------


def test_unknown_server_held_in_pending(tmp_path: Path) -> None:
    """First-time MCP server: nothing in the consent store → all its
    skills land in `_pending_mcp_manifests`, nothing admitted to context."""
    agent = _agent_with_home(tmp_path)
    manifests = [_mcp_manifest("alpha"), _mcp_manifest("beta")]

    admitted = agent._apply_consent_gate(manifests)

    assert admitted == []
    pending = agent.pending_skill_servers()
    assert "github" in pending
    assert {m.name for m in pending["github"]} == {"alpha", "beta"}


def test_approved_server_admitted_through_gate(tmp_path: Path) -> None:
    """A server whose stored fingerprint matches the discovered catalog
    is admitted directly — no manual approval needed mid-session."""
    consent_path = default_consent_path(tmp_path)
    manifests = [_mcp_manifest("alpha")]
    fingerprint = compute_catalog_fingerprint("github", manifests)
    SkillConsentStore(consent_path).approve("github", fingerprint)

    agent = _agent_with_home(tmp_path)
    admitted = agent._apply_consent_gate(manifests)

    assert {m.name for m in admitted} == {"alpha"}
    assert agent.pending_skill_servers() == {}


def test_approved_server_with_changed_catalog_held_pending(tmp_path: Path) -> None:
    """A server adding a new skill invalidates prior consent — the new
    catalog lands in pending, the old approval still on disk doesn't
    automatically admit it. This is the core anti-creep property."""
    consent_path = default_consent_path(tmp_path)
    original = [_mcp_manifest("alpha")]
    SkillConsentStore(consent_path).approve(
        "github", compute_catalog_fingerprint("github", original)
    )

    expanded = original + [_mcp_manifest("beta")]
    agent = _agent_with_home(tmp_path)
    admitted = agent._apply_consent_gate(expanded)

    assert admitted == []
    assert "github" in agent.pending_skill_servers()


def test_auto_approve_bypasses_prompt(tmp_path: Path) -> None:
    """`skills_auto_approve: true` writes consent through on first
    encounter and admits the catalog without a pending stage."""
    from fast_agent.config import MCPServerSettings

    agent = _agent_with_home(tmp_path)
    # Inject server config with auto-approve set.
    agent._context.config.mcp.servers["github"] = MCPServerSettings(
        skills_auto_approve=True
    )

    manifests = [_mcp_manifest("alpha")]
    admitted = agent._apply_consent_gate(manifests)

    assert {m.name for m in admitted} == {"alpha"}
    # And the consent store now has a record — subsequent sessions auto-admit too.
    store = SkillConsentStore(default_consent_path(tmp_path))
    assert store.is_approved("github", compute_catalog_fingerprint("github", manifests))


def test_gate_per_server_isolation(tmp_path: Path) -> None:
    """Approval for server A must not admit server B's skills.
    Mixed catalogs must be partitioned by server before fingerprinting."""
    consent_path = default_consent_path(tmp_path)
    github_skills = [_mcp_manifest("alpha", "github")]
    SkillConsentStore(consent_path).approve(
        "github", compute_catalog_fingerprint("github", github_skills)
    )

    acme_skills = [_mcp_manifest("zeta", "acme")]
    agent = _agent_with_home(tmp_path)
    admitted = agent._apply_consent_gate(github_skills + acme_skills)

    # github admitted; acme held pending.
    admitted_names = {m.name for m in admitted}
    assert admitted_names == {"alpha"}
    assert list(agent.pending_skill_servers().keys()) == ["acme"]


# --- approve / revoke ----------------------------------------------------


def test_approve_pending_server_admits_skills(tmp_path: Path) -> None:
    agent = _agent_with_home(tmp_path)
    manifests = [_mcp_manifest("alpha"), _mcp_manifest("beta")]
    agent._apply_consent_gate(manifests)

    assert agent.approve_skill_server("github") is True
    # Pending cleared; active set has the formerly-pending skills.
    assert agent.pending_skill_servers() == {}
    active_names = {m.name for m in agent._skill_manifests}
    assert {"alpha", "beta"}.issubset(active_names)


def test_approve_unknown_server_returns_false(tmp_path: Path) -> None:
    agent = _agent_with_home(tmp_path)
    # Nothing pending — caller should be told the action didn't apply
    # so a typo is distinguishable from a no-op (mirrors the disable
    # convention).
    assert agent.approve_skill_server("nonexistent") is False


def test_approve_persists_for_future_sessions(tmp_path: Path) -> None:
    """After approval, a fresh agent (same home) admits the same
    catalog without going through pending."""
    agent_1 = _agent_with_home(tmp_path)
    manifests = [_mcp_manifest("alpha")]
    agent_1._apply_consent_gate(manifests)
    assert agent_1.approve_skill_server("github") is True

    agent_2 = _agent_with_home(tmp_path)
    admitted = agent_2._apply_consent_gate(manifests)
    assert {m.name for m in admitted} == {"alpha"}
    assert agent_2.pending_skill_servers() == {}


def test_revoke_clears_pending(tmp_path: Path) -> None:
    agent = _agent_with_home(tmp_path)
    agent._apply_consent_gate([_mcp_manifest("alpha")])

    assert agent.revoke_skill_server("github") is True
    assert agent.pending_skill_servers() == {}


def test_revoke_clears_active_and_persisted(tmp_path: Path) -> None:
    """Revoke after approval: skills disappear from the active set AND
    the consent record is wiped. Next refresh will re-prompt."""
    agent = _agent_with_home(tmp_path)
    manifests = [_mcp_manifest("alpha")]
    agent._apply_consent_gate(manifests)
    agent.approve_skill_server("github")
    assert any(m.server_name == "github" for m in agent._skill_manifests)

    assert agent.revoke_skill_server("github") is True
    assert not any(m.server_name == "github" for m in agent._skill_manifests)

    store = SkillConsentStore(default_consent_path(tmp_path))
    assert not store.is_approved(
        "github", compute_catalog_fingerprint("github", manifests)
    )


def test_revoke_unknown_server_returns_false(tmp_path: Path) -> None:
    agent = _agent_with_home(tmp_path)
    assert agent.revoke_skill_server("nobody") is False


# --- safe defaults -------------------------------------------------------


def test_filesystem_skills_unaffected_by_gate(tmp_path: Path) -> None:
    """The gate operates only on MCP-served (URI-backed) manifests.
    Filesystem skills are user-installed and never gated — they're in
    `self._skill_manifests` before the gate runs."""
    fs_dir = tmp_path / "fs"
    fs_dir.mkdir()
    fs_md = fs_dir / "SKILL.md"
    fs_md.write_text(
        "---\nname: fs-skill\ndescription: filesystem\n---\nbody\n",
        encoding="utf-8",
    )
    fs_manifest = SkillManifest(
        name="fs-skill", description="filesystem", body="b", path=fs_md
    )

    config = AgentConfig(name="test", instruction="x", servers=[], skills=fs_dir)
    config.skill_manifests = [fs_manifest]
    ctx = Context()
    if ctx.config is None:
        from fast_agent.config import Settings

        ctx.config = Settings()
    ctx.config._fast_agent_home = str(tmp_path)
    agent = McpAgent(config=config, context=ctx)

    # Filesystem manifest is active before any MCP discovery runs.
    assert "fs-skill" in {m.name for m in agent._skill_manifests}
    # And the gate, given an MCP batch, doesn't touch the filesystem entry.
    agent._apply_consent_gate([_mcp_manifest("mcp-skill")])
    assert "fs-skill" in {m.name for m in agent._skill_manifests}
    assert "github" in agent.pending_skill_servers()


# --- template entries are part of the consent surface --------------------


def test_templates_from_pending_server_held_aside(tmp_path: Path) -> None:
    """A server publishing only `mcp-resource-template` entries can otherwise
    smuggle a manifest in via `/skills resolve` without ever being approved.
    Templates from pending servers must be partitioned to a held-aside set."""
    from fast_agent.mcp.mcp_skills_loader import SkillTemplateEntry

    agent = _agent_with_home(tmp_path)
    manifests = [_mcp_manifest("alpha")]
    template = SkillTemplateEntry(
        server_name="github",
        url_template="skill://docs/{product}/SKILL.md",
        description="docs",
    )
    agent._apply_consent_gate(manifests)
    active, pending = agent._partition_template_entries([template])

    assert active == []
    assert pending == {"github": [template]}


def test_templates_from_approved_server_are_active(tmp_path: Path) -> None:
    from fast_agent.mcp.mcp_skills_loader import SkillTemplateEntry

    consent_path = default_consent_path(tmp_path)
    manifests = [_mcp_manifest("alpha")]
    SkillConsentStore(consent_path).approve(
        "github", compute_catalog_fingerprint("github", manifests)
    )

    agent = _agent_with_home(tmp_path)
    agent._apply_consent_gate(manifests)
    template = SkillTemplateEntry(
        server_name="github",
        url_template="skill://docs/{product}/SKILL.md",
        description="docs",
    )
    active, pending = agent._partition_template_entries([template])

    assert active == [template]
    assert pending == {}


@pytest.mark.asyncio
async def test_register_resolved_template_rejects_pending_server(tmp_path: Path) -> None:
    """Defense-in-depth: even if a caller hands `register_resolved_skill_template`
    a template whose server is pending, the merge is refused. Without this,
    a non-UI code path could bypass `_partition_template_entries`."""
    from fast_agent.mcp.mcp_skills_loader import SkillTemplateEntry

    agent = _agent_with_home(tmp_path)
    # Put the server into the pending set without approval.
    agent._apply_consent_gate([_mcp_manifest("alpha")])
    assert "github" in agent.pending_skill_servers()

    template = SkillTemplateEntry(
        server_name="github",
        url_template="skill://docs/{product}/SKILL.md",
        description="docs",
    )

    manifest = await agent.register_resolved_skill_template(
        template, {"product": "anvil"}
    )

    assert manifest is None
    # Active manifests unchanged — pending server stays pending.
    assert not any(
        m.server_name == "github" for m in agent._skill_manifests
    )


def test_approve_lifts_pending_templates(tmp_path: Path) -> None:
    from fast_agent.mcp.mcp_skills_loader import SkillTemplateEntry

    agent = _agent_with_home(tmp_path)
    agent._apply_consent_gate([_mcp_manifest("alpha")])
    template = SkillTemplateEntry(
        server_name="github",
        url_template="skill://docs/{product}/SKILL.md",
        description="docs",
    )
    active, pending = agent._partition_template_entries([template])
    agent._skill_template_entries = active
    agent._pending_mcp_template_entries = pending
    assert agent._skill_template_entries == []
    assert agent._pending_mcp_template_entries == {"github": [template]}

    assert agent.approve_skill_server("github") is True
    # After approval, the template is active and the pending set is empty.
    assert template in agent._skill_template_entries
    assert agent._pending_mcp_template_entries == {}


def test_revoke_drops_active_templates(tmp_path: Path) -> None:
    from fast_agent.mcp.mcp_skills_loader import SkillTemplateEntry

    consent_path = default_consent_path(tmp_path)
    manifests = [_mcp_manifest("alpha")]
    SkillConsentStore(consent_path).approve(
        "github", compute_catalog_fingerprint("github", manifests)
    )
    agent = _agent_with_home(tmp_path)
    agent._apply_consent_gate(manifests)
    template = SkillTemplateEntry(
        server_name="github",
        url_template="skill://docs/{product}/SKILL.md",
        description="docs",
    )
    active, _ = agent._partition_template_entries([template])
    agent._skill_template_entries = active
    assert template in agent._skill_template_entries

    assert agent.revoke_skill_server("github") is True
    assert template not in agent._skill_template_entries
