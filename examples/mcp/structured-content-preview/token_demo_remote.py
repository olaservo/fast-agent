"""Remote variant of token_demo.py — measures content vs structuredContent token
cost against a LIVE deployed MCP server (Streamable HTTP).

Default target is the SEP-2200 **anti-pattern** server `customer-segmentation`,
whose `get-customer-data` tool returns a `content` text block that is a stringified
JSON DUPLICATE of its `structuredContent`. Because the two agree, the model's answer
is the same across all modes — the story here is purely token cost:

    - content-only    -> sees the stringified-JSON text block      (~baseline)
    - structured-wins -> sees the canonical structuredContent JSON  (~baseline)
    - both            -> sees BOTH, i.e. ~2x tokens for identical information

Requires ANTHROPIC_API_KEY. Free HF CPU Spaces cold-start, so the first run per
mode may be slow or need a retry. Run from this directory:
    uv run token_demo_remote.py

To try "the extreme" instead, point at the video_resource server (play_video
returns a base64 video — bunny-1mb alone is ~350K tokens, so expect huge counts
and possible context-window errors; that is the point of the example):
    SERVER, TOOL_PROMPT = "video_resource", "Call play_video with videoId 'bunny-1mb'."
"""

from __future__ import annotations

import asyncio
import os

from fast_agent import FastAgent

MODES = ["content-only", "structured-wins", "both"]

SERVER = "customer_segmentation"
PROMPT = (
    "Call get-customer-data for the Enterprise segment. Then, in one sentence, tell me "
    "how many Enterprise customers there are and the name of the single customer with "
    "the highest annualRevenue."
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
    os.environ["FAST_AGENT_TOOL_RESULT_SERIALIZATION"] = mode

    fast = FastAgent(
        f"token-demo-remote-{mode}",
        config_path="token_demo_remote.yaml",
        parse_cli_args=False,
        quiet=True,
    )
    captured: dict = {"mode": mode, "answer": "", "final_in": 0, "cum_in": 0, "cum_out": 0}

    @fast.agent(instruction=INSTRUCTION, servers=[SERVER])
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

    results = []
    for mode in MODES:
        print(f"[{mode}] connecting to live server '{SERVER}' (cold-start may be slow)...")
        results.append(await run_mode(mode))

    line = "=" * 100
    print("\n" + line)
    print(f"Live serialization comparison  (server: {SERVER}, tool: get-customer-data)")
    print("  content text is a stringified DUPLICATE of structuredContent -> answers agree, tokens differ")
    print(line)
    print(f"{'mode':<16}{'final-turn in':>14}{'cum in':>9}{'cum out':>9}   answer")
    print("-" * 100)
    for r in results:
        print(
            f"{r['mode']:<16}{r['final_in']:>14,}{r['cum_in']:>9,}{r['cum_out']:>9,}   "
            f"{r['answer'][:100]}"
        )
    print(line)
    base = next((r["final_in"] for r in results if r["mode"] == "structured-wins"), 0)
    both = next((r["final_in"] for r in results if r["mode"] == "both"), 0)
    if base and both:
        print(
            f"'both' final-turn input is {both - base:+,} tokens vs 'structured-wins' "
            f"({(both / base - 1) * 100:+.1f}%) — paid for a verbatim duplicate."
        )
    print()


if __name__ == "__main__":
    asyncio.run(main())
