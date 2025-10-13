# MCP-UI Integration Guide

## Overview

fast-agent provides comprehensive support for **MCP-UI**, a pattern for returning rich, interactive user interfaces alongside tool responses from MCP servers. This guide explains how MCP-UI works in fast-agent and how to use it effectively.

> **Official Documentation:** For basic setup and CLI usage, see the official [fast-agent MCP-UI documentation](https://fast-agent.ai/mcp/mcp-ui/).
>
> This guide provides **developer-focused details**, implementation examples, and a deep dive into building MCP servers with UI support.

## What is MCP-UI?

MCP-UI is a pattern built on top of the Model Context Protocol that allows MCP servers to return embedded UI resources (HTML, web applications, or remote DOM content) as part of their tool responses. These UI resources:

- Use the standard MCP `EmbeddedResource` content type
- Are identified by `ui://` URI scheme (a convention, not part of core MCP)
- Can provide interactive visualizations, dashboards, forms, or any web-based content
- Leverage MCP's existing resource and content mechanisms

MCP-UI is **not part of the official MCP specification** - it's an extension pattern that uses standard MCP features in a specific way to enable UI functionality.

## Architecture

fast-agent implements MCP-UI through a clean mixin pattern:

```
┌─────────────────┐
│  McpAgentWithUI │  ← Your agent class
└────────┬────────┘
         │
    ┌────┴──────────────────┐
    │                       │
┌───┴────┐          ┌──────┴──────┐
│McpUIMixin│         │  McpAgent   │
└────────┘          └─────────────┘
```

### Components

1. **McpAgentWithUI** (`src/fast_agent/mcp/ui_agent.py`)
   - Combines base `McpAgent` with UI functionality
   - Entry point for using MCP-UI features

2. **McpUIMixin** (`src/fast_agent/mcp/ui_mixin.py`)
   - Intercepts tool results to extract UI resources
   - Manages the `mcp-ui` message channel
   - Controls when and how UI resources are displayed

3. **UI Utilities** (`src/fast_agent/ui/mcp_ui_utils.py`)
   - Processes embedded resources by MIME type
   - Generates local HTML files with security sandboxing
   - Auto-opens UI resources in the browser

## How It Works

### 1. Tool Execution Flow

```
User Message
    ↓
Tool Calls (with MCP Server)
    ↓
Tool Results (containing ui:// resources)
    ↓
McpUIMixin.run_tools()
    ├─→ Extracts UI blocks
    ├─→ Removes them from tool results
    └─→ Stores in message.channels['mcp-ui']
    ↓
Assistant Message Display
    ↓
McpUIMixin.show_assistant_message()
    ├─→ Displays assistant text
    └─→ Processes UI resources from history
        ├─→ Generates HTML files in .fast-agent/ui/
        ├─→ Shows clickable links in console
        └─→ Auto-opens in browser (if mode="auto")
```

### 2. UI Resource Processing

When a tool returns a UI resource:

```python
# MCP Server returns:
EmbeddedResource(
    type="resource",
    resource=TextResourceContents(
        uri="ui://dashboard/analytics-123",
        mimeType="text/html",
        text="<html>...</html>"
    )
)
```

fast-agent processes it:
1. Detects `ui://` URI prefix
2. Extracts content based on MIME type
3. Creates a safe HTML wrapper with iframe sandbox
4. Writes to `.fast-agent/ui/dashboard_analytics-123.html`
5. Displays link and optionally opens in browser

### 3. Security

All HTML content is wrapped in a sandboxed iframe:

```html
<iframe
  src="..."
  sandbox="allow-scripts allow-forms allow-same-origin"
  referrerpolicy="no-referrer">
</iframe>
```

This prevents malicious code from accessing the parent context or making unauthorized requests.

## Supported MIME Types

### 1. `text/html`

Raw HTML content that will be rendered in an iframe.

**Example:**
```python
TextResourceContents(
    uri="ui://component/instance-1",
    mimeType="text/html",
    text="<html><body><h1>Dashboard</h1><div id='chart'></div></body></html>"
)
```

**Output:**
- File: `.fast-agent/ui/component_instance-1.html`
- Displayed as clickable link in console
- Auto-opened in browser (if `ui_mode="auto"`)

### 2. `text/uri-list`

A URL to an external web application.

**Example:**
```python
TextResourceContents(
    uri="ui://dashboard/external",
    mimeType="text/uri-list",
    text="https://example.com/dashboard?session=abc123"
)
```

**Output:**
- File: `.fast-agent/ui/dashboard_external.html` (iframe wrapper)
- Link opens the external URL
- Sandboxed iframe for security

### 3. `application/vnd.mcp-ui.remote-dom*`

Remote DOM resources (future support).

**Current Status:**
- Not fully supported yet
- Generates informational placeholder page
- Users are notified to upgrade when support is available

## Installation

For basic installation and setup, see the [official quick start guide](https://fast-agent.ai/mcp/mcp-ui/).

Quick installation:
```bash
# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install fast-agent
uv tool install -U fast-agent-mcp

# Set your API keys
export ANTHROPIC_API_KEY=your-key-here
# or OPENAI_API_KEY, etc.
```

### CLI Usage

Connect to an MCP-UI demo server using the CLI:

```bash
# Connect to a local MCP-UI server
fast-agent --url http://localhost:3000 --model=gpt-4o-mini

# Run with multiple models in parallel
fast-agent --url http://localhost:3000 --model=gpt-4o-mini,claude-3-5-sonnet-20241022

# Run with a specific prompt
fast-agent --url http://localhost:3000 --model=claude-3-5-haiku-20241022 -m "create a visualization"
```

## Configuration

### UI Modes

Control MCP-UI behavior with the `ui_mode` parameter:

```python
from fast_agent.mcp.ui_agent import McpAgentWithUI
from fast_agent.agents.agent_types import AgentConfig

agent = McpAgentWithUI(
    AgentConfig("my-agent", model="claude-3-5-sonnet-20241022", servers=["my-server"]),
    context=context,
    ui_mode="auto"  # or "enabled" or "disabled"
)
```

**Mode Options:**

| Mode | Behavior |
|------|----------|
| `"disabled"` | UI resources are not extracted or processed |
| `"enabled"` | UI resources are extracted and links displayed (no auto-open) |
| `"auto"` | UI resources extracted, links displayed, and auto-opened in browser |

### Global Configuration

In your `fastagent.config.yaml` (create with `fast-agent setup`):

```yaml
# Control MCP-UI mode globally
mcp_ui_mode: auto  # or "enabled" or "disabled"

# Customize output directory (relative to CWD or absolute)
mcp_ui_output_dir: ".fast-agent/ui"

# Optional: Configure servers with custom headers
mcp:
  servers:
    my-ui-server:
      transport: http
      url: https://example.com/mcp
      headers:
        Authorization: Bearer ${MY_TOKEN}
      # Optional: Client spoofing to adjust server behavior
      implementation:
        name: claude-code
        version: 1.0.99
```

For more configuration options, see the [official configuration documentation](https://fast-agent.ai/mcp/).

### Runtime Configuration

Change UI mode dynamically:

```python
agent.set_ui_mode("enabled")  # Disable auto-open
agent.set_ui_mode("disabled")  # Completely disable UI processing
```

## Usage Examples

### Basic Usage

```python
import asyncio
from fast_agent import FastAgent
from fast_agent.mcp.ui_agent import McpAgentWithUI
from fast_agent.agents.agent_types import AgentConfig

async def main():
    fast = FastAgent("MCP-UI Example")

    # Create agent with UI support
    config = AgentConfig(
        "ui-demo",
        model="claude-3-5-sonnet-20241022",
        servers=["my-visualization-server"]
    )

    agent = McpAgentWithUI(
        config,
        context=fast.context,
        ui_mode="auto"  # Auto-open UI resources
    )

    await agent.initialize()
    await agent.interactive()
    await fast.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### Programmatic Access

```python
async def analyze_with_ui():
    # Initialize agent
    agent = McpAgentWithUI(config, context=context, ui_mode="enabled")
    await agent.initialize()

    # Send message that triggers UI response
    response = await agent.generate([{
        "role": "user",
        "content": "Show me a visualization of the sales data"
    }])

    # UI resources are automatically processed
    # Check for UI content in previous message
    if len(agent.message_history) >= 2:
        prev_msg = agent.message_history[-2]
        if prev_msg.channels and 'mcp-ui' in prev_msg.channels:
            print(f"UI resources available: {len(prev_msg.channels['mcp-ui'])}")
```

### Custom Output Directory

```python
from fast_agent.config import get_settings

# Set custom output directory
settings = get_settings()
settings.mcp_ui_output_dir = "./my-ui-outputs"

# Now all UI files will be written to ./my-ui-outputs/
```

## Building MCP Servers with UI Support

If you're building an MCP server that returns UI resources:

### 1. Return UI Resources in Tool Results

```python
from mcp.types import CallToolResult, EmbeddedResource, TextResourceContents
from pydantic import AnyUrl

async def handle_tool_call(name: str, arguments: dict):
    # Your tool logic here
    data = process_data(arguments)

    # Generate visualization HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Visualization</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div id="chart"></div>
        <script>
            var data = {data};
            Plotly.newPlot('chart', data);
        </script>
    </body>
    </html>
    """

    # Return both text result and UI resource
    return CallToolResult(
        content=[
            {"type": "text", "text": "Here's the visualization of your data."},
            EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri=AnyUrl(f"ui://visualization/{arguments['id']}"),
                    mimeType="text/html",
                    text=html_content
                )
            )
        ]
    )
```

### 2. Use External URLs

For hosted dashboards:

```python
return CallToolResult(
    content=[
        {"type": "text", "text": "Dashboard is ready at the link below."},
        EmbeddedResource(
            type="resource",
            resource=TextResourceContents(
                uri=AnyUrl("ui://dashboard/session-123"),
                mimeType="text/uri-list",
                text="https://my-dashboard.com/session/abc123\n"
            )
        )
    ]
)
```

### 3. Best Practices

- **Unique URIs:** Use unique identifiers in `ui://` URIs to avoid filename collisions
- **Self-Contained HTML:** Include all CSS/JS inline or via CDN (no relative file references)
- **Responsive Design:** UI should work at various viewport sizes
- **Error Handling:** Provide fallback text content if UI can't be displayed
- **Security:** Sanitize any user input before including in HTML

## Output Files

### File Naming

UI resources are saved with sanitized filenames:

```
ui://component/instance-1  →  .fast-agent/ui/component_instance-1.html
ui://dashboard/analytics   →  .fast-agent/ui/dashboard_analytics.html
```

- Non-alphanumeric characters replaced with `_`
- Maximum filename length: 120 characters
- Duplicate names get numeric suffixes: `_1`, `_2`, etc.

### Directory Structure

```
.fast-agent/
└── ui/
    ├── component_instance-1.html
    ├── dashboard_analytics.html
    ├── chart_sales-2024.html
    └── ... (other UI files)
```

### Cleanup

The `.fast-agent/` directory is automatically added to `.gitignore`. You can safely delete UI files:

```bash
# Clean up all generated UI files
rm -rf .fast-agent/ui/

# They will be regenerated on next run
```

## Console Display

When UI resources are processed, fast-agent displays them in the console:

```
╭─ MCP-UI Resources ────────────────────────────────╮
│                                                    │
│  📊 component:instance-1                          │
│     file://.../component_instance-1.html          │
│                                                    │
│  📈 dashboard:analytics                           │
│     https://example.com/dashboard                 │
│                                                    │
╰────────────────────────────────────────────────────╯
```

In terminals with hyperlink support, these are clickable links.

## Troubleshooting

### UI Resources Not Appearing

1. **Check UI Mode:**
   ```python
   print(agent._ui_mode)  # Should be "auto" or "enabled"
   ```

2. **Verify Server Returns UI Resources:**
   - Check tool results contain `ui://` URIs
   - Verify MIME type is supported

3. **Check Message History:**
   ```python
   # UI resources stored in previous user message
   if agent.message_history:
       msg = agent.message_history[-2]  # Previous user message
       print(msg.channels.get('mcp-ui', []))
   ```

### Browser Not Opening

- **Mode Check:** Ensure `ui_mode="auto"`
- **Platform Issues:** Some platforms require manual opening
- **Permission Issues:** Browser may block file:// URLs in some configurations
- **Solution:** Manually open files from `.fast-agent/ui/` directory

### File Not Found Errors

- **Directory Creation:** Ensure write permissions in current directory
- **Custom Directory:** Check `mcp_ui_output_dir` setting is valid
- **Path Issues:** Use absolute paths for `mcp_ui_output_dir` if needed

### Security Warnings

Some browsers warn about local HTML files:
- This is expected for `file://` URLs
- UI content is sandboxed for security
- You can also serve files via local HTTP server if needed

## Advanced Topics

### Custom UI Processing

Extend the UI processing by subclassing:

```python
from fast_agent.mcp.ui_mixin import McpUIMixin

class CustomUIMixin(McpUIMixin):
    async def _display_ui_resources(self, resources):
        # Custom display logic
        for resource in resources:
            # Process resource
            pass

        # Call parent implementation
        await super()._display_ui_resources(resources)

class CustomUIAgent(CustomUIMixin, McpAgent):
    pass
```

### Serving UI via HTTP

For better browser compatibility:

```python
import http.server
import socketserver
import threading
from pathlib import Path

def start_ui_server(port=8000):
    """Start local HTTP server for UI files."""
    ui_dir = Path(".fast-agent/ui")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ui_dir), **kwargs)

    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"Serving UI at http://localhost:{port}")
        httpd.serve_forever()

# Run in background thread
threading.Thread(target=start_ui_server, daemon=True).start()
```

### Integration with FastAPI

Serve MCP-UI resources via FastAPI:

```python
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI()

@app.get("/ui/{filename}")
async def serve_ui(filename: str):
    ui_dir = Path(".fast-agent/ui")
    file_path = ui_dir / filename

    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)

    return {"error": "File not found"}, 404
```

## Testing

fast-agent includes comprehensive tests for MCP-UI:

```bash
# Run unit tests
pytest tests/unit/fast_agent/mcp/test_ui_mixin.py

# Run integration tests
pytest tests/integration/mcp_ui/test_mcp_ui_integration.py
```

### Writing Tests for Your MCP Server

```python
import pytest
from mcp.types import EmbeddedResource, TextResourceContents
from pydantic import AnyUrl

@pytest.mark.asyncio
async def test_ui_resource_generation():
    # Test that your server returns UI resources
    result = await your_tool_handler("visualize", {"data": [1, 2, 3]})

    # Check for UI resource
    ui_resources = [
        item for item in result.content
        if isinstance(item, EmbeddedResource)
    ]

    assert len(ui_resources) > 0
    assert str(ui_resources[0].resource.uri).startswith("ui://")
```

## Resources

### Official Documentation
- **fast-agent MCP-UI docs:** https://fast-agent.ai/mcp/mcp-ui/
- **fast-agent main docs:** https://fast-agent.ai
- **MCP Configuration:** https://fast-agent.ai/mcp/

### Protocol Specifications
- **MCP Protocol Specification:** https://modelcontextprotocol.io/
- **Note:** MCP-UI is a pattern/extension built on top of MCP, not part of the core specification

### Source Code
- **Implementation:**
  - `src/fast_agent/mcp/ui_agent.py` - McpAgentWithUI class
  - `src/fast_agent/mcp/ui_mixin.py` - Core UI extraction logic
  - `src/fast_agent/ui/mcp_ui_utils.py` - UI processing utilities
- **Examples:** `examples/mcp/mcp-ui/`
- **Tests:**
  - `tests/unit/fast_agent/mcp/test_ui_mixin.py`
  - `tests/integration/mcp_ui/test_mcp_ui_integration.py`

## Contributing

If you're building MCP servers with UI support or have improvements to the MCP-UI implementation:

1. Test with fast-agent's comprehensive MCP-UI support
2. Share your examples in the examples directory
3. Report issues or suggest improvements at https://github.com/evalstate/fast-agent/issues

## Future Enhancements

Planned improvements to MCP-UI support:

- [ ] Full `application/vnd.mcp-ui.remote-dom` support
- [ ] Built-in UI server mode (no external HTTP server needed)
- [ ] UI state persistence across sessions
- [ ] Better mobile/responsive display
- [ ] UI thumbnail previews in console
- [ ] Real-time UI updates via WebSocket

---

**Note:** MCP-UI is an evolving specification. fast-agent aims to provide the most complete and robust implementation available. Please report any issues or compatibility concerns.
