import io

from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax

from fast_agent.ui.markdown_renderables import (
    build_markdown_renderable,
    extract_single_fenced_code_block,
)


def test_build_markdown_renderable_uses_syntax_for_code_only_fence() -> None:
    renderable = build_markdown_renderable(
        "```bash\necho hi\n```",
        code_theme="monokai",
        escape_xml=True,
    )

    assert isinstance(renderable, Syntax)

    output = io.StringIO()
    Console(file=output, force_terminal=False, width=40).print(renderable)
    rendered = output.getvalue().splitlines()
    assert any(line.startswith("echo hi") for line in rendered)


def test_build_markdown_renderable_keeps_mixed_markdown_as_markdown() -> None:
    renderable = build_markdown_renderable(
        "Run this:\n\n```python\nprint(1)\n```",
        code_theme="monokai",
        escape_xml=True,
    )

    assert isinstance(renderable, Markdown)


def test_extract_single_fenced_code_block_handles_incomplete_stream() -> None:
    block = extract_single_fenced_code_block("```python\nprint('hi')")

    assert block is not None
    assert block.language == "python"
    assert block.code == "print('hi')"
    assert block.complete is False
