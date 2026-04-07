from __future__ import annotations

import re
from dataclasses import dataclass

from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.text import Text

from fast_agent.ui.markdown_helpers import prepare_markdown_content

_FENCE_OPEN_LINE_RE = re.compile(r"^\s{0,3}(?P<delim>`{3,}|~{3,})(?P<info>.*)$")


@dataclass(frozen=True)
class FencedCodeBlock:
    language: str
    code: str
    complete: bool


def extract_single_fenced_code_block(text: str) -> FencedCodeBlock | None:
    if not text:
        return None

    lines = text.splitlines()
    if not lines:
        return None

    start_index = 0
    while start_index < len(lines) and not lines[start_index].strip():
        start_index += 1
    if start_index >= len(lines):
        return None

    end_index = len(lines) - 1
    while end_index >= start_index and not lines[end_index].strip():
        end_index -= 1

    opening = _FENCE_OPEN_LINE_RE.match(lines[start_index])
    if opening is None:
        return None

    delimiter = opening.group("delim")
    info = opening.group("info")
    if delimiter[0] == "`" and "`" in info:
        return None

    fence_char = delimiter[0]
    fence_len = len(delimiter)
    language = info.strip().split(" ", 1)[0] or "text"

    closing_index: int | None = None
    for index in range(start_index + 1, end_index + 1):
        line = lines[index]
        stripped = line.lstrip(" ")
        if len(line) - len(stripped) > 3:
            continue
        if not stripped or stripped[0] != fence_char:
            continue

        marker_len = 0
        while marker_len < len(stripped) and stripped[marker_len] == fence_char:
            marker_len += 1
        if marker_len >= fence_len and stripped[marker_len:].strip() == "":
            closing_index = index
            break

    if closing_index is None:
        return FencedCodeBlock(
            language=language,
            code="\n".join(lines[start_index + 1 :]),
            complete=False,
        )

    if closing_index != end_index:
        return None

    return FencedCodeBlock(
        language=language,
        code="\n".join(lines[start_index + 1 : closing_index]),
        complete=True,
    )


def close_incomplete_code_blocks(text: str) -> str:
    if "```" not in text and "~~~" not in text:
        return text

    in_fence = False
    fence_char = "`"
    fence_len = 3

    for line in text.splitlines():
        stripped = line.lstrip(" ")
        if len(line) - len(stripped) > 3:
            continue

        if not in_fence:
            opening = _FENCE_OPEN_LINE_RE.match(line)
            if opening is None:
                continue

            delimiter = opening.group("delim")
            info = opening.group("info")
            if delimiter[0] == "`" and "`" in info:
                continue

            in_fence = True
            fence_char = delimiter[0]
            fence_len = len(delimiter)
            continue

        if not stripped or stripped[0] != fence_char:
            continue

        marker_len = 0
        while marker_len < len(stripped) and stripped[marker_len] == fence_char:
            marker_len += 1
        if marker_len >= fence_len and stripped[marker_len:].strip() == "":
            in_fence = False

    if not in_fence:
        return text

    closing_fence = fence_char * fence_len
    if text.endswith("\n"):
        return f"{text}{closing_fence}\n"
    return f"{text}\n{closing_fence}\n"


def build_markdown_renderable(
    text: str,
    *,
    code_theme: str,
    escape_xml: bool,
    cursor_suffix: str = "",
    close_incomplete_fences: bool = False,
):
    if not text and not cursor_suffix:
        return Text("")

    code_block = extract_single_fenced_code_block(text)
    if code_block is not None:
        code = code_block.code
        if cursor_suffix:
            code += cursor_suffix
        return Syntax(
            code,
            code_block.language,
            theme=code_theme,
            line_numbers=False,
            word_wrap=False,
        )

    prepared = prepare_markdown_content(text, escape_xml)
    if close_incomplete_fences:
        prepared = close_incomplete_code_blocks(prepared)
    if cursor_suffix:
        prepared += cursor_suffix
    if not prepared:
        return Text("")
    return Markdown(prepared, code_theme=code_theme)


__all__ = [
    "FencedCodeBlock",
    "build_markdown_renderable",
    "close_incomplete_code_blocks",
    "extract_single_fenced_code_block",
]
