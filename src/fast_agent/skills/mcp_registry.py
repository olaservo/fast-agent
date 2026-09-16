"""Verified Skills-over-MCP registry and resource-set installer."""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import re
import shutil
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Protocol
from urllib.parse import unquote, urlsplit

import frontmatter
from mcp_types import BlobResourceContents, ServerCapabilities, TextResourceContents

from fast_agent.core.logging.logger import get_logger
from fast_agent.marketplace import git_sources as marketplace_git_sources
from fast_agent.mcp.skills_extension import DYNAMIC_RESOURCES
from fast_agent.skills.models import (
    SKILL_NAME_PATTERN,
    SKILL_SOURCE_FILENAME,
    McpSkillResource,
)
from fast_agent.skills.provenance import (
    build_mcp_installed_skill_source,
    compute_skill_content_fingerprint,
    read_installed_skill_source,
    write_installed_skill_source,
)
from fast_agent.skills.registry import SkillRegistry

if TYPE_CHECKING:
    from mcp.client import CacheMode
    from mcp_types import ReadResourceResult

    from fast_agent.mcp.skills_extension import (
        GetSkillResult,
        ListSkillsResult,
        SkillEntry,
        SkillResource,
    )


class McpSkillRegistryClient(Protocol):
    async def get_capabilities(self, server_name: str) -> ServerCapabilities | None: ...

    async def list_skills(self, server_name: str, cursor: str | None) -> "ListSkillsResult": ...

    async def get_skill(self, uri: str, server_name: str) -> "GetSkillResult": ...


class McpSkillInstallClient(McpSkillRegistryClient, Protocol):
    async def get_resource(
        self,
        resource_uri: str,
        *,
        server_name: str | None = None,
        cache_mode: "CacheMode" = "use",
    ) -> "ReadResourceResult": ...


logger = get_logger(__name__)

SKILLS_EXTENSION = "io.modelcontextprotocol/skills"
MAX_LIST_PAGES = 1_000
MAX_LIST_ENTRIES = 10_000
# SEP-2640 "Limits": every conforming host MUST accept a skill up to these bounds, and
# a server SHOULD NOT serve one that exceeds them. Both are checkable from the entry
# alone (count the resources, sum their sizes) before any file is fetched.
MAX_SKILL_RESOURCES = 512
MAX_SKILL_BYTES = 16 * 1_048_576
# A single file can legitimately be the whole 16 MiB budget.
MAX_RESOURCE_BYTES = MAX_SKILL_BYTES
MAX_SKILL_MD_BYTES = 262_144
MAX_SERVER_SKILL_BYTES = 200 * 1_048_576
MAX_RESOURCE_PATH_LENGTH = 1_024
MAX_INSTALL_DIR_SERVER_SEGMENT = 48
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_INSTALL_DIR_SEPARATOR = "--"
_UNSAFE_DIR_CHARS_RE = re.compile(r"[^a-z0-9]+")
_ENCODED_SEPARATOR_RE = re.compile(r"%(?:2f|5c)", re.IGNORECASE)
_WINDOWS_INVALID_CHARS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)


@dataclass(frozen=True)
class McpRegistrySkill:
    """A skill entry as served by one MCP server.

    ``resources`` is the complete, digest-bearing manifest of the skill's files, or
    ``None`` when the server declared ``"resources": "dynamic"``: a generated skill
    that publishes no stable digests and therefore cannot be verified or installed.
    ``limit_violation`` records why a well-formed entry exceeds the SEP-2640 per-skill
    limits; such a skill stays listed so the user can see it, but cannot be installed.
    """

    name: str
    description: str
    uri: str
    server_name: str
    server_version: str | None = None
    frontmatter: dict[str, Any] = field(default_factory=dict)
    resources: tuple["SkillResource", ...] | None = None
    limit_violation: str | None = None

    @property
    def source_url(self) -> str:
        return self.uri

    @property
    def is_dynamic(self) -> bool:
        return self.resources is None

    @property
    def total_size(self) -> int | None:
        if self.resources is None:
            return None
        return sum(resource.size for resource in self.resources)

    @property
    def install_blocker(self) -> str | None:
        """Why this skill cannot be installed, or ``None`` if it can."""
        if self.is_dynamic:
            return (
                f"MCP skill {self.uri} publishes dynamic content ('resources': 'dynamic'); "
                "it has no digests to verify against and cannot be installed"
            )
        return self.limit_violation

    @property
    def revision(self) -> str | None:
        # The revision is the content-bound identity SEP-2640 ties approvals to: the
        # set of (uri, digest) pairs. `size` is deliberately excluded so that a server
        # adding sizes to an unchanged skill does not present as an update.
        if self.resources is None:
            return None
        payload = sorted(
            ({"uri": resource.uri, "digest": resource.digest} for resource in self.resources),
            key=lambda item: item["uri"],
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    @property
    def install_dir_name(self) -> str:
        return mcp_skill_install_dir_name(self.server_name, self.name)


def mcp_skill_install_dir_name(server_name: str, skill_name: str) -> str:
    """Directory name under which an MCP-served skill is materialized.

    SEP-2640 makes (server, URI) the identity of a skill and requires every path at
    which a host materializes skill content to encode the server identity, so that
    same-named skills from different servers land at distinct paths and the origin
    is recoverable from the path. The name is ``<server>--<skill>``: the host's own
    label for the server, lowercased and reduced to ``[a-z0-9-]``, followed by the
    skill's ``name``. When the server label is not already in that alphabet (or is
    long) a short hash of the exact label is appended so distinct labels never
    collapse onto one directory. The skill name itself is left untouched, so the
    final segment still reads as the Agent Skills ``name``.
    """
    _validate_skill_name(skill_name)
    label = server_name.strip()
    if not label:
        raise ValueError("MCP server name is required to name the install directory")
    folded = _UNSAFE_DIR_CHARS_RE.sub("-", label.casefold()).strip("-")
    lossless = folded == label and len(folded) <= MAX_INSTALL_DIR_SERVER_SEGMENT
    if not lossless:
        suffix = hashlib.sha256(label.encode()).hexdigest()[:8]
        folded = f"{folded[: MAX_INSTALL_DIR_SERVER_SEGMENT - len(suffix) - 1].strip('-')}-{suffix}"
        folded = folded.lstrip("-")
    return f"{folded}{_INSTALL_DIR_SEPARATOR}{skill_name}"


@dataclass(frozen=True)
class McpSkillRegistry:
    server_name: str
    server_version: str | None
    skills: list[McpRegistrySkill]

    @property
    def display_name(self) -> str:
        version = f"@{self.server_version}" if self.server_version else ""
        return f"mcp-server {self.server_name}{version}"


def _extension_settings(capabilities: ServerCapabilities | None) -> Mapping[str, Any] | None:
    if capabilities is None or not isinstance(capabilities.extensions, Mapping):
        return None
    settings = capabilities.extensions.get(SKILLS_EXTENSION)
    return settings if isinstance(settings, Mapping) else None


def server_supports_mcp_skills(capabilities: ServerCapabilities | None) -> bool:
    return _extension_settings(capabilities) is not None


def server_supports_directory_read(capabilities: ServerCapabilities | None) -> bool:
    """Whether the server declared the optional directory-read method."""
    settings = _extension_settings(capabilities)
    return settings is not None and settings.get("directoryRead") is True


async def scan_mcp_skill_registry(
    aggregator: McpSkillRegistryClient,
    server_name: str,
    *,
    server_version: str | None = None,
) -> McpSkillRegistry | None:
    try:
        capabilities = await aggregator.get_capabilities(server_name)
    except Exception as exc:
        logger.debug(
            "MCP skills capability unavailable", data={"server": server_name, "error": str(exc)}
        )
        return None
    if not server_supports_mcp_skills(capabilities):
        return None

    try:
        entries = await _list_skill_entries(aggregator, server_name)
        skills = [
            _registry_skill(entry, server_name=server_name, server_version=server_version)
            for entry in entries
        ]
        _reject_duplicate_uris(skills)
    except Exception as exc:
        logger.warning(
            "MCP skills/list failed validation", data={"server": server_name, "error": str(exc)}
        )
        skills = []
    return McpSkillRegistry(server_name=server_name, server_version=server_version, skills=skills)


async def get_mcp_registry_skill(
    aggregator: McpSkillRegistryClient,
    uri: str,
    server_name: str,
    *,
    server_version: str | None = None,
) -> McpRegistrySkill:
    result = await aggregator.get_skill(uri, server_name)
    entry = result.skill
    skill = _registry_skill(entry, server_name=server_name, server_version=server_version)
    if skill.uri != uri:
        raise ValueError(f"skills/get returned {skill.uri!r}, not requested URI {uri!r}")
    return skill


async def _list_skill_entries(
    aggregator: McpSkillRegistryClient, server_name: str
) -> list["SkillEntry"]:
    entries: list[SkillEntry] = []
    seen_cursors: set[str] = set()
    cursor: str | None = None
    for _ in range(MAX_LIST_PAGES):
        result = await aggregator.list_skills(server_name, cursor)
        page = result.skills
        if not isinstance(page, list):
            raise ValueError("skills/list returned a non-list skills field")
        entries.extend(page)
        if len(entries) > MAX_LIST_ENTRIES:
            raise ValueError("skills/list exceeds entry limit")
        cursor = result.next_cursor
        if cursor is None:
            return entries
        if not isinstance(cursor, str) or cursor in seen_cursors:
            raise ValueError("skills/list returned an invalid or repeated cursor")
        seen_cursors.add(cursor)
    raise ValueError("skills/list exceeds page limit")


def _registry_skill(
    entry: "SkillEntry", *, server_name: str, server_version: str | None
) -> McpRegistrySkill:
    frontmatter_value = entry.frontmatter
    if not isinstance(frontmatter_value, Mapping):
        raise ValueError("skill frontmatter must be an object")
    frontmatter = dict(frontmatter_value)
    name = _required_frontmatter_string(frontmatter, "name")
    description = _required_frontmatter_string(frontmatter, "description")
    _validate_skill_name(name)
    uri = entry.uri
    root = _skill_root(uri, name)
    resources_value = entry.resources
    limit_violation: str | None = None
    if resources_value == DYNAMIC_RESOURCES:
        resources = None
    else:
        if not isinstance(resources_value, (list, tuple)):
            raise ValueError("skill resources must be a list or the string 'dynamic'")
        resources = tuple(resources_value)
        _validate_resource_set(uri, root, resources)
        limit_violation = _limit_violation(uri, resources)
        if limit_violation is not None:
            logger.warning(
                "MCP skill exceeds SEP-2640 per-skill limits and cannot be installed",
                data={"server": server_name, "uri": uri, "reason": limit_violation},
            )
    return McpRegistrySkill(
        name=name,
        description=description,
        uri=uri,
        server_name=server_name,
        server_version=server_version,
        frontmatter=frontmatter,
        resources=resources,
        limit_violation=limit_violation,
    )


def _limit_violation(uri: str, resources: tuple["SkillResource", ...]) -> str | None:
    """The SEP-2640 per-skill limit this entry breaks, judged from the entry alone."""
    count = len(resources)
    if count > MAX_SKILL_RESOURCES:
        return (
            f"MCP skill {uri} lists {count} resources, more than the "
            f"{MAX_SKILL_RESOURCES} allowed per skill"
        )
    total = sum(resource.size for resource in resources)
    if total > MAX_SKILL_BYTES:
        return (
            f"MCP skill {uri} totals {total:,} bytes, more than the "
            f"{MAX_SKILL_BYTES:,} bytes ({MAX_SKILL_BYTES // 1_048_576} MiB) allowed per skill"
        )
    return None


def _required_frontmatter_string(frontmatter: Mapping[str, Any], field: str) -> str:
    value = frontmatter.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"skill frontmatter requires nonempty {field}")
    return value


def _skill_root(uri: str, name: str) -> tuple[str, str, str]:
    parsed = _split_safe_uri(uri)
    if not parsed.path.endswith("/SKILL.md"):
        raise ValueError("skill URI must end exactly with /SKILL.md")
    parent = parsed.path[: -len("/SKILL.md")]
    segments = [segment for segment in (parsed.netloc, *parent.split("/")) if segment]
    if not segments or segments[-1] != name:
        raise ValueError("skill URI final path segment must equal its name")
    return parsed.scheme, parsed.netloc, parent


def _split_safe_uri(uri: object):
    if not isinstance(uri, str) or not uri:
        raise ValueError("skill URI is required")
    parsed = urlsplit(uri)
    if not parsed.scheme or parsed.scheme.casefold() == "file":
        raise ValueError("skill URI must be a non-file absolute URI")
    if (
        parsed.query
        or parsed.fragment
        or "\\" in parsed.path
        or _ENCODED_SEPARATOR_RE.search(parsed.path)
    ):
        raise ValueError("skill URI contains unsafe path syntax")
    return parsed


def _validate_resource_set(
    skill_uri: str, root: tuple[str, str, str], resources: tuple["SkillResource", ...]
) -> None:
    if not resources:
        raise ValueError("skill resources must be a complete nonempty list")
    seen_uris: set[str] = set()
    seen_paths: dict[str, str] = {}
    top_level = 0
    for resource in resources:
        uri = resource.uri
        digest = resource.digest
        if not isinstance(uri, str) or uri in seen_uris:
            raise ValueError("skill resources contain duplicate or invalid URIs")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError("skill resources require lowercase SHA256 digests")
        size = resource.size
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("skill resources require a non-negative integer size")
        seen_uris.add(uri)
        relative = _resource_relative_path(uri, root)
        if relative.casefold() == SKILL_SOURCE_FILENAME.casefold():
            raise ValueError(f"skill resources cannot use reserved path {SKILL_SOURCE_FILENAME}")
        key = unicodedata.normalize("NFC", relative).casefold()
        if key in seen_paths:
            raise ValueError("skill resource paths collide under case/Unicode normalization")
        seen_paths[key] = uri
        if uri == skill_uri:
            top_level += 1
    if top_level != 1:
        raise ValueError("skill resources must contain exactly one top-level SKILL.md URI")


def _resource_relative_path(uri: str, root: tuple[str, str, str]) -> str:
    parsed = _split_safe_uri(uri)
    scheme, netloc, root_path = root
    if (parsed.scheme, parsed.netloc) != (scheme, netloc):
        raise ValueError("skill resource is outside the skill URI root")
    prefix = f"{root_path}/" if root_path else "/"
    if not parsed.path.startswith(prefix):
        raise ValueError("skill resource is outside the skill URI root")
    raw_relative = parsed.path[len(prefix) :]
    if not raw_relative:
        raise ValueError("skill resource path is empty")
    decoded = unquote(raw_relative)
    if len(decoded) > MAX_RESOURCE_PATH_LENGTH:
        raise ValueError("skill resource path exceeds length limit")
    parts = decoded.split("/")
    if any(not _is_portable_path_segment(part) for part in parts):
        raise ValueError("skill resource path is unsafe")
    path = PurePosixPath(*parts)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("skill resource path is unsafe")
    return path.as_posix()


def _is_portable_path_segment(segment: str) -> bool:
    if (
        not segment
        or segment in {".", ".."}
        or segment.endswith((" ", "."))
        or any(ord(char) < 32 or char in _WINDOWS_INVALID_CHARS for char in segment)
    ):
        return False
    return segment.split(".", 1)[0].casefold() not in _WINDOWS_RESERVED_NAMES


def _validate_skill_name(name: str) -> None:
    if not SKILL_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "skill name must be 1-64 lowercase letters, numbers, or hyphens and "
            "must not start or end with a hyphen"
        )
    if name.casefold() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("skill name is reserved by the local filesystem")


def _reject_duplicate_uris(skills: Iterable[McpRegistrySkill]) -> None:
    seen: set[str] = set()
    for skill in skills:
        if skill.uri in seen:
            raise ValueError("skills/list contains duplicate skill URIs")
        seen.add(skill.uri)


async def install_mcp_registry_skill(
    aggregator: McpSkillInstallClient,
    skill: McpRegistrySkill,
    *,
    destination_root: Path,
) -> Path:
    blocker = skill.install_blocker
    if blocker is not None:
        raise ValueError(blocker)
    dir_name = skill.install_dir_name
    install_dir = destination_root.resolve() / dir_name
    if install_dir.exists():
        raise FileExistsError(f"Skill already exists: {install_dir}")
    destination_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination_root, prefix=f".{dir_name}.install-") as temp:
        staged_dir = Path(temp) / dir_name
        fresh = await _refresh_before_fetch(aggregator, skill)
        await _stage_verified_mcp_skill(aggregator, fresh, staged_dir, managed_dir=destination_root)
        staged_dir.rename(install_dir)
    return install_dir


async def update_mcp_registry_skill(
    aggregator: McpSkillInstallClient,
    skill: McpRegistrySkill,
    *,
    skill_dir: Path,
) -> Path:
    skill_dir = skill_dir.resolve()
    with tempfile.TemporaryDirectory(
        dir=skill_dir.parent, prefix=f".{skill_dir.name}.update-"
    ) as temp:
        staged_dir = Path(temp) / skill_dir.name
        fresh = await _refresh_before_fetch(aggregator, skill)
        await _stage_verified_mcp_skill(
            aggregator,
            fresh,
            staged_dir,
            managed_dir=skill_dir.parent,
            exclude=skill_dir,
        )
        marketplace_git_sources.atomic_replace_directory(
            existing_dir=skill_dir, staged_dir=staged_dir
        )
    return skill_dir


async def _refresh_before_fetch(
    aggregator: McpSkillInstallClient, skill: McpRegistrySkill
) -> McpRegistrySkill:
    fresh = await get_mcp_registry_skill(
        aggregator,
        skill.uri,
        skill.server_name,
        server_version=skill.server_version,
    )
    blocker = fresh.install_blocker
    if blocker is not None:
        raise ValueError(blocker)
    if fresh.revision != skill.revision or _canonical_frontmatter(
        fresh.frontmatter
    ) != _canonical_frontmatter(skill.frontmatter):
        raise ValueError("MCP skill changed since it was selected; refresh and try again")
    return fresh


async def _stage_verified_mcp_skill(
    aggregator: McpSkillInstallClient,
    skill: McpRegistrySkill,
    install_dir: Path,
    *,
    managed_dir: Path,
    exclude: Path | None = None,
) -> None:
    assert skill.resources is not None
    files: list[tuple[str, bytes]] = []
    total = 0
    root = _skill_root(skill.uri, skill.name)
    for resource in skill.resources:
        result = await aggregator.get_resource(
            resource.uri,
            server_name=skill.server_name,
            cache_mode="refresh",
        )
        content = _resource_bytes(result, resource.uri)
        if len(content) > MAX_RESOURCE_BYTES:
            raise ValueError(f"MCP resource exceeds per-file limit: {resource.uri}")
        total += len(content)
        if total > MAX_SKILL_BYTES:
            raise ValueError("MCP skill resource set exceeds total-size limit")
        # SEP-2640: a read whose byte length differs from the entry's `size` is a
        # verification failure equivalent to a digest mismatch, hashed or not.
        if len(content) != resource.size:
            raise ValueError(
                f"MCP resource size mismatch: {resource.uri} "
                f"(expected {resource.size} bytes, read {len(content)})"
            )
        digest = f"sha256:{hashlib.sha256(content).hexdigest()}"
        if digest != resource.digest:
            raise ValueError(f"MCP resource SHA256 mismatch: {resource.uri}")
        files.append((_resource_relative_path(resource.uri, root), content))

    install_dir.mkdir(parents=True, exist_ok=False)
    try:
        for relative, content in files:
            destination = install_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        _verify_manifest(skill, install_dir)
        _strip_permission_widening_frontmatter(install_dir)
        fingerprint = compute_skill_content_fingerprint(install_dir)
        resources = tuple(
            McpSkillResource(uri=resource.uri, digest=resource.digest)
            for resource in skill.resources
        )
        revision = skill.revision
        assert revision is not None
        write_installed_skill_source(
            install_dir,
            build_mcp_installed_skill_source(
                server_name=skill.server_name,
                server_version=skill.server_version,
                skill_uri=skill.uri,
                fingerprint=fingerprint,
                resources=resources,
                revision=revision,
            ),
        )
        _check_server_budget(managed_dir, skill.server_name, _directory_size(install_dir), exclude)
    except Exception:
        shutil.rmtree(install_dir, ignore_errors=True)
        raise


def _resource_bytes(result: "ReadResourceResult", requested_uri: str) -> bytes:
    if len(result.contents) != 1:
        raise ValueError(f"MCP resource returned ambiguous content: {requested_uri}")
    item = result.contents[0]
    if str(item.uri) != requested_uri:
        raise ValueError(f"MCP resource returned content for the wrong URI: {requested_uri}")
    if isinstance(item, TextResourceContents):
        return item.text.encode()
    if isinstance(item, BlobResourceContents):
        if len(item.blob) > ((MAX_RESOURCE_BYTES + 2) // 3) * 4:
            raise ValueError(f"MCP resource exceeds per-file limit: {requested_uri}")
        return base64.b64decode(item.blob, validate=True)
    raise ValueError(f"MCP resource returned unsupported content: {requested_uri}")


def _verify_manifest(skill: McpRegistrySkill, install_dir: Path) -> None:
    manifest_path = install_dir / "SKILL.md"
    if not manifest_path.is_file():
        raise ValueError("MCP skill resource set must contain root SKILL.md")
    content = manifest_path.read_bytes()
    if len(content) > MAX_SKILL_MD_BYTES:
        raise ValueError("MCP skill SKILL.md exceeds size limit")
    text = content.decode("utf-8")
    manifest, error = SkillRegistry.parse_manifest_text(text)
    if manifest is None:
        raise ValueError(f"Failed to parse MCP skill manifest: {error}")
    if manifest.name != skill.name:
        raise ValueError("MCP skill manifest name does not match the resource manifest")
    served = frontmatter.loads(text).metadata or {}
    if _canonical_frontmatter(served) != _canonical_frontmatter(skill.frontmatter):
        raise ValueError("MCP skill frontmatter does not match the resource manifest")


def _canonical_frontmatter(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("MCP skill frontmatter object keys must be strings")
        return (
            "object",
            tuple(sorted((key, _canonical_frontmatter(item)) for key, item in value.items())),
        )
    if isinstance(value, (list, tuple)):
        return ("array", tuple(_canonical_frontmatter(item) for item in value))
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return ("string", value.isoformat())
    if value is None:
        return ("null", None)
    if isinstance(value, bool):
        return ("boolean", value)
    if isinstance(value, int):
        return ("integer", value)
    if isinstance(value, float):
        return ("number", value)
    if isinstance(value, str):
        return ("string", value)
    raise ValueError(f"Unsupported MCP skill frontmatter value: {type(value).__name__}")


_PERMISSION_WIDENING_FRONTMATTER = ("allowed-tools", "hooks")


def _strip_permission_widening_frontmatter(install_dir: Path) -> None:
    manifest = install_dir / "SKILL.md"
    post = frontmatter.loads(manifest.read_text(encoding="utf-8"))
    removed = [field for field in _PERMISSION_WIDENING_FRONTMATTER if field in post.metadata]
    if not removed:
        return
    for key in removed:
        post.metadata.pop(key)
    manifest.write_text(frontmatter.dumps(post), encoding="utf-8")
    logger.warning("Stripped MCP skill permission frontmatter", data={"skill": install_dir.name})


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _check_server_budget(
    managed_dir: Path, server_name: str, projected: int, exclude: Path | None
) -> None:
    used = 0
    if managed_dir.is_dir():
        for child in managed_dir.iterdir():
            if not child.is_dir() or child == exclude:
                continue
            source = read_installed_skill_source(child).source
            if (
                source is not None
                and source.source_origin == "mcp"
                and source.mcp_server_name == server_name
            ):
                used += _directory_size(child)
    if used + projected > MAX_SERVER_SKILL_BYTES:
        raise ValueError(f"MCP server '{server_name}' exceeds cumulative skill-size limit")


def select_mcp_registry_skill(
    entries: Iterable[McpRegistrySkill], selector: str
) -> McpRegistrySkill | None:
    value = selector.strip()
    if not value:
        return None
    listed = list(entries)
    if value.isdigit():
        index = int(value)
        return listed[index - 1] if 1 <= index <= len(listed) else None
    exact_uri = [entry for entry in listed if entry.uri == value]
    if exact_uri:
        return exact_uri[0]
    matches = [entry for entry in listed if entry.name.casefold() == value.casefold()]
    if len(matches) > 1:
        raise LookupError(f"Skill name is ambiguous: {value}")
    return matches[0] if matches else None


