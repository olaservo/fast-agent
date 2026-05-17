"""Tests for the `<mcp-skill-content>` wrapper on MCP skill reads.

The wrapper is a navigational/audit marker: it records which server
returned the body and the URI it came from, so transcripts and logs
can be traced back to source. SEP-2640 recommends a stronger "treat
as untrusted" framing; this host diverges and treats MCP skill
content the same way it treats tool descriptions and MCP
`prompts/get` — server-authored text gated at connect time, not
per-skill. The wrapper carries no security framing for the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcp.types import ReadResourceResult, TextResourceContents
from pydantic import AnyUrl

from fast_agent.skills.registry import SkillManifest, format_skills_for_prompt
from fast_agent.tools.skill_reader import SkillReader


def _text_result(text: str, uri: str) -> ReadResourceResult:
    return ReadResourceResult(
        contents=[TextResourceContents(uri=AnyUrl(uri), mimeType="text/markdown", text=text)]
    )


def _mcp_manifest(name: str = "git-workflow", server: str = "srv") -> SkillManifest:
    return SkillManifest(
        name=name,
        description=f"The {name} skill",
        body="",
        path=None,
        uri=f"skill://{name}/SKILL.md",
        server_name=server,
    )


def _fake_aggregator(responses: dict[str, ReadResourceResult | Exception]) -> Any:
    agg = MagicMock()

    async def get_resource(uri: str, *, server_name: str | None = None) -> ReadResourceResult:
        result = responses.get(uri)
        if result is None:
            raise ValueError(f"unknown uri {uri}")
        if isinstance(result, Exception):
            raise result
        return result

    agg.get_resource = get_resource
    return agg


# --- MCP path wraps ------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregator_read_is_wrapped_with_server_and_uri() -> None:
    manifest = _mcp_manifest("git-workflow", server="github")
    agg = _fake_aggregator(
        {"skill://git-workflow/SKILL.md": _text_result("# body", "skill://git-workflow/SKILL.md")}
    )
    reader = SkillReader([manifest], logger=MagicMock(), aggregator=agg)

    result = await reader.execute({"path": "skill://git-workflow/SKILL.md"})

    assert not result.isError
    text = result.content[0].text
    # Wrapper records server name and source URI for transcripts/audit.
    # Bare server name — element name already conveys "MCP-served."
    assert text.startswith(
        '<mcp-skill-content server="github" '
        'uri="skill://git-workflow/SKILL.md">'
    )
    assert "# body" in text
    assert text.rstrip().endswith("</mcp-skill-content>")


@pytest.mark.asyncio
async def test_unenumerated_uri_wraps_with_unknown_server() -> None:
    """The unenumerated `skill://` fanout path doesn't know which server
    answered. The wrapper still fires, marking the server as `(unknown)` —
    a strictly weaker but still-honest attribution. Critically, the
    wrapper is *not* skipped just because we lack a name."""
    agg = _fake_aggregator(
        {"skill://surprise/SKILL.md": _text_result("# body", "skill://surprise/SKILL.md")}
    )
    reader = SkillReader([], logger=MagicMock(), aggregator=agg)

    result = await reader.execute({"path": "skill://surprise/SKILL.md"})

    assert not result.isError
    text = result.content[0].text
    assert 'server="(unknown)"' in text
    assert "# body" in text


# --- Filesystem path does NOT wrap ---------------------------------------


@pytest.mark.asyncio
async def test_filesystem_read_is_not_wrapped(tmp_path: Path) -> None:
    """Filesystem skills have no server to attribute and no source URI
    to record, so the wrapper has nothing to mark. Leaving filesystem
    reads unwrapped keeps the wrapper a precise marker for the MCP
    boundary rather than ambient noise."""
    skill_dir = tmp_path / "alpha"
    skill_dir.mkdir()
    md = skill_dir / "SKILL.md"
    md.write_text("---\nname: alpha\ndescription: x\n---\n# alpha body\n", encoding="utf-8")
    manifest = SkillManifest(name="alpha", description="x", body="b", path=md)
    reader = SkillReader([manifest], logger=MagicMock())

    result = await reader.execute({"path": str(md)})

    assert not result.isError
    text = result.content[0].text
    assert "mcp-skill-content" not in text
    assert "# alpha body" in text


# --- Preamble does NOT lecture the model about the wrapper ---------------


def test_preamble_does_not_lecture_on_trust() -> None:
    """The wrapper is an audit/navigation marker, not a security
    instruction. The preamble must not coach the model to distrust
    wrapped content — that framing would create a trust gradient
    between MCP skills and tool descriptions/prompts (which cross the
    same trust boundary unannotated)."""
    manifest = _mcp_manifest("git-workflow", server="github")
    out = format_skills_for_prompt([manifest])
    lowered = out.lower()
    # No alarm words anywhere in the preamble.
    assert "untrusted" not in lowered
    assert "not as authoritative" not in lowered
    assert "reference material" not in lowered
    # Mechanical guidance about URI handling is still expected — the
    # preamble must teach the model HOW to read MCP-served skills,
    # just not lecture about whether to trust them.
    assert "uri" in lowered
    assert "skill://" in out


def test_preamble_mentions_source_element_when_mcp_skill_present() -> None:
    """SEP-2640 SHOULD: hosts indicate which server an MCP-served skill
    came from. The preamble points the model at the `<source>` element
    so it can do that without prose lecturing."""
    manifest = _mcp_manifest("git-workflow", server="github")
    out = format_skills_for_prompt([manifest])
    assert "<source>" in out


def test_preamble_omits_mcp_guidance_when_only_filesystem_skills(tmp_path: Path) -> None:
    """If no MCP skill is present, the URI/source guidance is dead
    weight — don't include it. Symmetric with the existing
    has_mcp_skill flag."""
    md = tmp_path / "alpha" / "SKILL.md"
    md.parent.mkdir(parents=True)
    md.write_text("---\nname: alpha\ndescription: x\n---\nbody\n", encoding="utf-8")
    manifest = SkillManifest(name="alpha", description="x", body="b", path=md)
    out = format_skills_for_prompt([manifest])
    assert "mcp-skill-content" not in out
    assert "skill://" not in out
    assert "<source>" not in out
