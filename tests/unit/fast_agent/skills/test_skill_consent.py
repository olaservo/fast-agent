"""Tests for SkillConsentStore: per-server skill-catalog approval."""

from __future__ import annotations

from pathlib import Path

import pytest

from fast_agent.skills.consent import (
    SkillConsentStore,
    compute_catalog_fingerprint,
    default_consent_path,
)
from fast_agent.skills.registry import SkillManifest


def _mcp_manifest(name: str, server: str, uri: str | None = None) -> SkillManifest:
    return SkillManifest(
        name=name,
        description=f"The {name} skill",
        body="",
        path=None,
        uri=uri or f"skill://{name}/SKILL.md",
        server_name=server,
    )


# --- fingerprint ---------------------------------------------------------


def test_fingerprint_is_deterministic_over_ordering() -> None:
    """Same catalog, different list order, same hash. Sessions discovering
    the same skills in any order must compute the same fingerprint —
    otherwise consent would be order-dependent."""
    a = _mcp_manifest("alpha", "github")
    b = _mcp_manifest("beta", "github")
    c = _mcp_manifest("gamma", "github")

    fp_1 = compute_catalog_fingerprint("github", [a, b, c])
    fp_2 = compute_catalog_fingerprint("github", [c, a, b])
    assert fp_1 == fp_2


def test_fingerprint_filters_by_server() -> None:
    """Manifests from a different server are ignored. The fingerprint
    is server-scoped: server X's consent must not match a catalog that
    happens to contain skills from server Y."""
    g1 = _mcp_manifest("alpha", "github")
    a1 = _mcp_manifest("alpha", "acme")
    fp_github = compute_catalog_fingerprint("github", [g1, a1])
    fp_acme = compute_catalog_fingerprint("acme", [g1, a1])
    # Different inputs → different hashes.
    assert fp_github != fp_acme


def test_fingerprint_changes_when_skill_added() -> None:
    """A server adding a new skill must invalidate prior consent.
    Otherwise an approved server could silently sneak in a new skill."""
    base = [_mcp_manifest("alpha", "github")]
    expanded = base + [_mcp_manifest("beta", "github")]
    assert compute_catalog_fingerprint("github", base) != compute_catalog_fingerprint(
        "github", expanded
    )


def test_fingerprint_changes_when_uri_rebound() -> None:
    """Same name, different URI = a different skill behind the name.
    Fingerprint must catch this — otherwise a server could swap a skill
    body by re-pointing the URI without re-prompting."""
    original = [_mcp_manifest("alpha", "github", "skill://alpha/SKILL.md")]
    swapped = [_mcp_manifest("alpha", "github", "skill://alpha-evil/SKILL.md")]
    assert compute_catalog_fingerprint(
        "github", original
    ) != compute_catalog_fingerprint("github", swapped)


# --- store: load / save / round-trip --------------------------------------


def test_missing_file_is_empty(tmp_path: Path) -> None:
    """Absent consent file is the safe default — no one is approved."""
    store = SkillConsentStore(tmp_path / "skill_consent.json")
    assert store.all_records() == {}
    assert not store.is_approved("github", "any-fingerprint")


def test_approve_persists_across_instances(tmp_path: Path) -> None:
    path = tmp_path / "skill_consent.json"
    s1 = SkillConsentStore(path)
    s1.approve("github", "fp-1")

    s2 = SkillConsentStore(path)
    assert s2.is_approved("github", "fp-1")
    assert s2.stored_fingerprint("github") == "fp-1"


def test_revoke_persists(tmp_path: Path) -> None:
    path = tmp_path / "skill_consent.json"
    s1 = SkillConsentStore(path)
    s1.approve("github", "fp-1")
    assert s1.revoke("github") is True

    s2 = SkillConsentStore(path)
    assert not s2.is_approved("github", "fp-1")
    assert s2.all_records() == {}


def test_revoke_unknown_is_noop(tmp_path: Path) -> None:
    """Revoking a server that was never approved returns False and
    doesn't touch the file — caller can distinguish typo from no-op."""
    store = SkillConsentStore(tmp_path / "skill_consent.json")
    assert store.revoke("nonexistent") is False


def test_is_approved_only_for_matching_fingerprint(tmp_path: Path) -> None:
    """The approval is bound to a specific fingerprint. A different
    fingerprint for the same server name returns False — this is the
    core mechanism for "catalog changed → re-prompt"."""
    store = SkillConsentStore(tmp_path / "skill_consent.json")
    store.approve("github", "fp-1")
    assert store.is_approved("github", "fp-1")
    assert not store.is_approved("github", "fp-2-different")


def test_corrupt_file_treated_as_empty(tmp_path: Path) -> None:
    """A malformed consent file must not crash startup. The safe
    behavior is "no one is approved" until the user re-approves —
    refusing to load is preferable to silently auto-approving."""
    path = tmp_path / "skill_consent.json"
    path.write_text("{ this is not json", encoding="utf-8")

    store = SkillConsentStore(path)
    assert store.all_records() == {}
    assert not store.is_approved("github", "anything")


def test_partial_record_skipped(tmp_path: Path) -> None:
    """Records missing required fields are dropped silently. Don't crash
    on a hand-edited or partially-written file — just ignore the
    incomplete entries."""
    path = tmp_path / "skill_consent.json"
    path.write_text(
        '{"github": {"fingerprint": "fp-1", "approved_at": 100}, '
        '"acme": {"fingerprint": "no_timestamp"}, '
        '"bad": "not-a-dict"}',
        encoding="utf-8",
    )
    store = SkillConsentStore(path)
    records = store.all_records()
    assert "github" in records
    assert "acme" not in records
    assert "bad" not in records


# --- default path resolution ---------------------------------------------


def test_default_consent_path_uses_home(tmp_path: Path) -> None:
    p = default_consent_path(tmp_path)
    assert p == tmp_path / "skill_consent.json"


def test_default_consent_path_without_home() -> None:
    """No-home fallback writes under `~/.fast-agent/` rather than cwd —
    avoids littering arbitrary directories from headless runs."""
    p = default_consent_path(None)
    assert p.name == "skill_consent.json"
    assert ".fast-agent" in p.parts
