"""Local wire models for the SEP-2640 (Final) Skills extension.

The pinned MCP SDK does not yet provide Skills Extension request/result types.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp_types import (
    CacheableResult,
    ListResourcesResult,
    PaginatedRequestParams,
    PaginatedResult,
    Request,
    RequestParams,
    Result,
)
from pydantic import BaseModel, ConfigDict, Field, StrictInt

DYNAMIC_RESOURCES: Literal["dynamic"] = "dynamic"
"""Marker a server uses in place of a resource manifest for generated skills."""


class SkillResource(BaseModel):
    """A content-addressed resource belonging to a skill.

    ``size`` is the byte length of the raw content the ``digest`` covers. SEP-2640
    requires it on every entry so hosts can budget a skill before fetching anything.
    """

    model_config = ConfigDict(populate_by_name=True)

    uri: str = Field(alias="uri")
    digest: str = Field(alias="digest")
    size: StrictInt = Field(alias="size", ge=0)


class SkillEntry(BaseModel):
    """A skill's metadata and its complete resource manifest.

    ``resources`` is required: either the complete list of the skill's files or the
    literal string ``"dynamic"`` for generated skills that cannot publish stable
    digests. An entry with ``resources`` missing, or of any other shape, is invalid.
    """

    model_config = ConfigDict(populate_by_name=True)

    uri: str = Field(alias="uri")
    frontmatter: dict[str, Any] = Field(alias="frontmatter")
    resources: list[SkillResource] | Literal["dynamic"] = Field(alias="resources")

    @property
    def is_dynamic(self) -> bool:
        return self.resources == DYNAMIC_RESOURCES


class ListSkillsRequestParams(PaginatedRequestParams):
    """Parameters for the ``skills/list`` request."""


class ListSkillsRequest(Request[ListSkillsRequestParams, Literal["skills/list"]]):
    """Request for the SEP-2640 skills listing."""

    method: Literal["skills/list"] = "skills/list"
    params: ListSkillsRequestParams


class ListSkillsResult(PaginatedResult, CacheableResult):
    """The paginated, cacheable response to ``skills/list``."""

    skills: list[Annotated[SkillEntry | dict[str, Any], Field(union_mode="left_to_right")]] = Field(
        alias="skills"
    )
    """Entries as served. One that fails ``SkillEntry`` validation is kept as the raw
    object rather than failing the whole page, so a single invalid entry (which a host
    MUST NOT load) does not hide every other skill the server publishes."""
    result_type: Literal["complete"] = Field(default="complete", alias="resultType")


class GetSkillRequestParams(RequestParams):
    """Parameters for the ``skills/get`` request."""

    uri: str = Field(alias="uri")


class GetSkillRequest(Request[GetSkillRequestParams, Literal["skills/get"]]):
    """Request for a single SEP-2640 skill entry."""

    method: Literal["skills/get"] = "skills/get"
    params: GetSkillRequestParams


class GetSkillResult(Result):
    """The response to ``skills/get``."""

    skill: SkillEntry = Field(alias="skill")
    result_type: Literal["complete"] = Field(default="complete", alias="resultType")


class DirectoryReadRequestParams(PaginatedRequestParams):
    """Parameters for the SEP-2640 directory-read extension."""

    uri: str = Field(alias="uri")


class DirectoryReadRequest(
    Request[DirectoryReadRequestParams, Literal["resources/directory/read"]]
):
    """Request for the SEP-2640 directory-read extension."""

    method: Literal["resources/directory/read"] = "resources/directory/read"
    params: DirectoryReadRequestParams


class DirectoryReadResult(ListResourcesResult):
    """The response to ``resources/directory/read``.

    The direct children of the directory, with subdirectories listed as
    ``inode/directory`` resources. The shape is ``resources/list``'s, including
    the base protocol's ``resultType``, which the SDK's ``ListResourcesResult``
    already declares.
    """
