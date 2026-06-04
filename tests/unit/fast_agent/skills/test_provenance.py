from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.skills.models import (
    SKILL_SOURCE_SCHEMA_VERSION,
    InstalledSkillSource,
    MarketplaceSkill,
)
from fast_agent.skills.provenance import (
    build_installed_skill_source,
    compute_skill_content_fingerprint,
    detect_skill_drift,
    read_installed_skill_source,
    write_installed_skill_source,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_installed_skill_source_round_trip(tmp_path: Path) -> None:
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()

    source = InstalledSkillSource(
        schema_version=SKILL_SOURCE_SCHEMA_VERSION,
        installed_via="marketplace",
        source_origin="remote",
        repo_url="https://github.com/example/skills",
        repo_ref="main",
        repo_path="skills/alpha",
        source_url="https://github.com/example/skills/blob/main/skills/alpha/SKILL.md",
        installed_commit="abcdef1234567890",
        installed_path_oid="feedbeef",
        installed_revision="abcdef1234567890",
        installed_at="2026-03-10T12:00:00Z",
        content_fingerprint="sha256:test",
    )

    write_installed_skill_source(skill_dir, source)
    loaded, error = read_installed_skill_source(skill_dir)

    assert error is None
    assert loaded == source


def _write_mcp_source(skill_dir: Path, *, fingerprint: str) -> None:
    write_installed_skill_source(
        skill_dir,
        InstalledSkillSource(
            schema_version=SKILL_SOURCE_SCHEMA_VERSION,
            installed_via="mcp",
            source_origin="mcp",
            repo_url="mcp://example-server",
            repo_ref=None,
            repo_path="skill://alpha/SKILL.md",
            source_url="skill://alpha/SKILL.md",
            installed_commit=None,
            installed_path_oid=None,
            installed_revision="sha256:" + "a" * 64,
            installed_at="2026-06-02T12:00:00Z",
            content_fingerprint=fingerprint,
            mcp_server_name="example-server",
            artifact_digest="sha256:" + "a" * 64,
            artifact_type="skill-md",
        ),
    )


def test_detect_skill_drift_clean(tmp_path: Path) -> None:
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: alpha\n---\nbody\n", encoding="utf-8")
    _write_mcp_source(skill_dir, fingerprint=compute_skill_content_fingerprint(skill_dir))

    assert detect_skill_drift(skill_dir) == "clean"


def test_detect_skill_drift_drifted(tmp_path: Path) -> None:
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: alpha\n---\nbody\n", encoding="utf-8")
    _write_mcp_source(skill_dir, fingerprint=compute_skill_content_fingerprint(skill_dir))

    # Locally modify the cached content after recording its fingerprint.
    (skill_dir / "SKILL.md").write_text("---\nname: alpha\n---\ntampered\n", encoding="utf-8")

    assert detect_skill_drift(skill_dir) == "drifted"


def test_detect_skill_drift_unknown_when_unmanaged(tmp_path: Path) -> None:
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: alpha\n---\nbody\n", encoding="utf-8")

    assert detect_skill_drift(skill_dir) == "unknown"


def test_build_installed_skill_source_uses_local_revision_without_commit() -> None:
    source = build_installed_skill_source(
        skill=MarketplaceSkill(
            name="alpha",
            description="demo",
            repo_url="/tmp/example-skills",
            repo_ref=None,
            repo_path="skills/alpha",
        ),
        source_origin="local",
        installed_commit=None,
        installed_path_oid=None,
        fingerprint="sha256:test",
    )

    assert source.schema_version == SKILL_SOURCE_SCHEMA_VERSION
    assert source.installed_revision == "local"
    assert source.installed_at.endswith("Z")
