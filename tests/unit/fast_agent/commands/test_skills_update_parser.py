import os
from pathlib import Path

from fast_agent.commands.handlers.skills import _format_update_results, _parse_update_argument
from fast_agent.skills.models import (
    SKILL_SOURCE_FILENAME,
    InstalledSkillSource,
    SkillUpdateInfo,
)
from fast_agent.skills.provenance import (
    DRIFT_WARNING_DETAIL,
    compute_skill_content_fingerprint,
    write_installed_skill_source,
)


def test_parse_update_argument_dry_run() -> None:
    selector, force, yes, error = _parse_update_argument(None)
    assert selector is None
    assert force is False
    assert yes is False
    assert error is None


def test_parse_update_argument_with_flags_and_selector() -> None:
    selector, force, yes, error = _parse_update_argument("--force alpha")
    assert selector == "alpha"
    assert force is True
    assert yes is False
    assert error is None


def test_parse_update_argument_all_with_yes() -> None:
    selector, force, yes, error = _parse_update_argument("all --yes")
    assert selector == "all"
    assert force is False
    assert yes is True
    assert error is None


def test_parse_update_argument_rejects_unknown_option() -> None:
    selector, force, yes, error = _parse_update_argument("--bogus")
    assert selector is None
    assert force is False
    assert yes is False
    assert error == "Unknown option: --bogus"


def test_format_update_results_shows_short_revision_and_installed_time() -> None:
    source = InstalledSkillSource(
        schema_version=1,
        installed_via="marketplace",
        source_origin="remote",
        repo_url="https://github.com/example/skills",
        repo_ref="main",
        repo_path="skills/demo",
        source_url=None,
        installed_commit="abcdef1234567890",
        installed_path_oid="deadbeef",
        installed_revision="abcdef1234567890",
        installed_at="2026-02-13T23:11:29Z",
        content_fingerprint="sha256:test",
    )
    updates = [
        SkillUpdateInfo(
            index=1,
            name="demo",
            skill_dir=Path("/tmp/demo"),
            status="update_available",
            current_revision="abcdef1234567890",
            available_revision="1234567890abcdef",
            managed_source=source,
        )
    ]

    rendered = _format_update_results(updates, title="Skill update check:").plain

    assert "source: /tmp/demo" in rendered
    assert "provenance: https://github.com/example/skills@main (skills/demo)" in rendered
    assert "revision: abcdef1 -> 1234567" in rendered
    assert "installed: 2026-02-13 23:11:29" in rendered
    assert "status: update available" in rendered


def test_format_update_results_reports_local_drift(tmp_path: Path) -> None:
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\nbody\n", encoding="utf-8")
    source = InstalledSkillSource(
        schema_version=1,
        installed_via="marketplace",
        source_origin="remote",
        repo_url="https://github.com/example/skills",
        repo_ref="main",
        repo_path="skills/demo",
        source_url=None,
        installed_commit="abcdef1234567890",
        installed_path_oid="deadbeef",
        installed_revision="abcdef1234567890",
        installed_at="2026-02-13T23:11:29Z",
        content_fingerprint=compute_skill_content_fingerprint(skill_dir),
    )
    write_installed_skill_source(skill_dir, source)

    # Tamper with the cached content after recording its fingerprint, ensuring
    # its mtime is clearly newer than the sidecar so the drift check engages.
    (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\ntampered\n", encoding="utf-8")
    newer = (skill_dir / SKILL_SOURCE_FILENAME).stat().st_mtime + 1000
    os.utime(skill_dir / "SKILL.md", (newer, newer))

    updates = [
        SkillUpdateInfo(
            index=1,
            name="demo",
            skill_dir=skill_dir,
            status="update_available",
            managed_source=source,
        )
    ]

    rendered = _format_update_results(updates, title="Skill update check:").plain

    assert f"drift: {DRIFT_WARNING_DETAIL}" in rendered


def test_format_update_results_uses_markdown_safe_detail_prefix() -> None:
    updates = [
        SkillUpdateInfo(
            index=1,
            name="demo",
            skill_dir=Path("/tmp/demo"),
            status="up_to_date",
        )
    ]

    rendered = _format_update_results(updates, title="Skill update check:").plain

    assert "  - source: /tmp/demo" in rendered
    assert "  - provenance:" in rendered
    assert "\n     source:" not in rendered


def test_format_update_results_shows_error_status_text() -> None:
    updates = [
        SkillUpdateInfo(
            index=1,
            name="broken",
            skill_dir=Path("/tmp/broken"),
            status="source_unreachable",
            detail="git ls-remote failed",
        )
    ]

    rendered = _format_update_results(updates, title="Skill update check:").plain

    assert "source_unreachable" not in rendered
    assert "status: source unreachable: git ls-remote failed" in rendered


def test_format_update_results_unmanaged_omits_redundant_status_line() -> None:
    updates = [
        SkillUpdateInfo(
            index=1,
            name="local-skill",
            skill_dir=Path("/tmp/local-skill"),
            status="unmanaged",
            detail="no sidecar metadata",
        )
    ]

    rendered = _format_update_results(updates, title="Skill update check:").plain

    assert "provenance: unmanaged." in rendered
    assert "status: unmanaged" not in rendered
    assert "unmanaged: no sidecar metadata" not in rendered
