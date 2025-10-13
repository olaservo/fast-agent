"""
Simple MCP-UI Demo Agent

This example demonstrates how to use fast-agent's MCP-UI support
to handle interactive UI resources from MCP servers.

Usage:
    uv run simple_ui_agent.py
"""

import asyncio

from fast_agent import FastAgent
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.mcp.ui_agent import McpAgentWithUI

# Create the FastAgent application
fast = FastAgent("MCP-UI Demo")


async def main():
    """Run the MCP-UI demo agent."""
    print("=" * 60)
    print("MCP-UI Demo Agent")
    print("=" * 60)
    print()
    print("This agent demonstrates MCP-UI support in fast-agent.")
    print("When an MCP server returns UI resources (ui:// URIs),")
    print("they will be automatically:")
    print("  1. Extracted from tool results")
    print("  2. Saved as HTML files in .fast-agent/ui/")
    print("  3. Displayed as clickable links in the console")
    print("  4. Auto-opened in your browser (in 'auto' mode)")
    print()
    print("=" * 60)
    print()

    # Configure the agent with MCP-UI support
    # Replace 'your-mcp-server' with an actual MCP server that returns UI resources
    config = AgentConfig(
        name="ui-demo",
        model="claude-3-5-sonnet-20241022",
        instruction="""You are a helpful assistant that works with MCP servers
        that may return interactive visualizations and UI components.
        When you receive UI resources, explain what they show.""",
        servers=[],  # Add your MCP servers here
    )

    # Create agent with UI support
    # ui_mode options:
    #   - "auto": Extract UI and auto-open in browser
    #   - "enabled": Extract UI and show links (no auto-open)
    #   - "disabled": Disable UI processing
    agent = McpAgentWithUI(config, context=fast.context, ui_mode="auto")

    await agent.initialize()

    # Print UI mode information
    print(f"UI Mode: {agent._ui_mode}")
    print(f"UI Output Directory: {fast.context.config.mcp_ui_output_dir}")
    print()

    # Display available commands
    print("Commands:")
    print("  - Type your messages normally")
    print("  - Type 'mode:enabled' to disable auto-open")
    print("  - Type 'mode:auto' to enable auto-open")
    print("  - Type 'mode:disabled' to disable UI processing")
    print("  - Type 'exit' or 'quit' to exit")
    print()

    # Start interactive session
    await agent.interactive()

    await fast.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
