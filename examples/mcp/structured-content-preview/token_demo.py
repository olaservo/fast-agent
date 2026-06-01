"""Demo: how `content` vs `structuredContent` serialization affects tokens + answers.

Runs the SAME tool-calling prompt against a real Anthropic model under each of
fast-agent's three tool-result serialization modes, then prints a comparison of
token usage and the model's answer.

The tool used is `structured_content_mismatch` from preview_server.py, whose
`content` text and `structuredContent` deliberately DISAGREE about ticket T-100:
    - content text         -> status "closed"
    - structuredContent    -> status "open"

So the model's answer is a direct readout of which field(s) it actually saw:
    - content-only    -> "closed"   (structuredContent ignored, classic-SDK behavior)
    - structured-wins -> "open"     (content text dropped, fast-agent default)
    - both            -> sees the conflict (and pays for both in input tokens)

Requires ANTHROPIC_API_KEY. Run from this directory:
    uv run token_demo.py
"""

from __future__ import annotations

import asyncio
import os

from fast_agent import FastAgent

MODES = ["content-only", "structured-wins", "both"]

PROMPT = (
    "Call the structured_content_mismatch tool. Then, in one short sentence, tell me "
    "the current status of ticket T-100 according to the tool result. If the result "
    "contains conflicting statuses for T-100, say so explicitly."
)

INSTRUCTION = (
    "You are a helpful assistant. When asked, call the requested MCP tool and answer "
    "strictly based on the tool result you receive."
)


def _tokens(usage) -> tuple[int, int, int]:
    """Return (final-turn input, cumulative input, cumulative output) tokens."""
    if usage is None or not getattr(usage, "turns", None):
        return (0, 0, 0)
    final_in = int(getattr(usage.turns[-1], "input_tokens", 0) or 0)
    summary = usage.get_summary()
    cum_in = int(summary.get("cumulative_input_tokens") or 0)
    cum_out = int(summary.get("cumulative_output_tokens") or 0)
    return (final_in, cum_in, cum_out)


async def run_mode(mode: str) -> dict:
    # The canonicalizer reads this env var at call time (it takes precedence over
    # the cached settings singleton), so each run gets a clean, independent mode.
    os.environ["FAST_AGENT_TOOL_RESULT_SERIALIZATION"] = mode

    fast = FastAgent(
        f"token-demo-{mode}",
        config_path="token_demo.yaml",
        parse_cli_args=False,
        quiet=True,
    )
    captured: dict = {"mode": mode, "answer": "", "final_in": 0, "cum_in": 0, "cum_out": 0}

    @fast.agent(instruction=INSTRUCTION, servers=["structured_preview"])
    async def _run() -> None:
        async with fast.run() as app:
            answer = await app.send(PROMPT)
            final_in, cum_in, cum_out = _tokens(app._agent(None).usage_accumulator)
            captured.update(
                answer=" ".join(answer.split()),
                final_in=final_in,
                cum_in=cum_in,
                cum_out=cum_out,
            )

    await _run()
    return captured


async def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set; this demo needs a real model.")

    results = [await run_mode(mode) for mode in MODES]

    line = "=" * 92
    print("\n" + line)
    print("Tool-result serialization comparison  (tool: structured_content_mismatch)")
    print("  T-100 in content text = 'closed'   |   T-100 in structuredContent = 'open'")
    print(line)
    print(f"{'mode':<16}{'final-turn in':>14}{'cum in':>9}{'cum out':>9}   answer")
    print("-" * 92)
    for r in results:
        print(
            f"{r['mode']:<16}{r['final_in']:>14,}{r['cum_in']:>9,}{r['cum_out']:>9,}   "
            f"{r['answer'][:96]}"
        )
    print(line)
    base = next((r["final_in"] for r in results if r["mode"] == "structured-wins"), 0)
    both = next((r["final_in"] for r in results if r["mode"] == "both"), 0)
    if base and both:
        print(
            f"'both' final-turn input is {both - base:+,} tokens vs 'structured-wins' "
            f"({(both / base - 1) * 100:+.1f}%) — the cost of duplicating the payload."
        )
    print()


if __name__ == "__main__":
    asyncio.run(main())
