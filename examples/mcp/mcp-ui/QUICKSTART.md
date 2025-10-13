# MCP-UI Quick Start Guide

Get started with MCP-UI in fast-agent in under 5 minutes!

> **Note:** This guide focuses on the examples in this directory. For CLI usage and connecting to existing MCP-UI servers, see the [official fast-agent MCP-UI documentation](https://fast-agent.ai/mcp/mcp-ui/).

## Prerequisites

1. Install fast-agent:
   ```bash
   uv tool install -U fast-agent-mcp
   ```

2. Install MCP Python SDK (for the mock server):
   ```bash
   uv pip install mcp
   ```

3. Set your API key:
   ```bash
   export ANTHROPIC_API_KEY=your-key-here
   # or OPENAI_API_KEY, etc.
   ```

## Quick Start (3 Steps)

### 1. Clone or Navigate to the MCP-UI Example

```bash
cd examples/mcp/mcp-ui
```

### 2. Test the Mock UI Server

First, verify the mock server works:

```bash
# Test the mock server directly (optional)
uv run mock_ui_server.py
# Press Ctrl+C after you see it running
```

### 3. Run the Demo Agent

```bash
uv run simple_ui_agent.py
```

## Try It Out

Once the agent starts, try these example prompts:

### Example 1: Simple Bar Chart
```
Create a bar chart showing monthly sales:
January: 100, February: 150, March: 120, April: 180, May: 200
```

**What happens:**
- Agent calls `create_chart` tool
- Server returns HTML visualization
- fast-agent saves it to `.fast-agent/ui/`
- Browser automatically opens showing interactive chart

### Example 2: Analytics Dashboard
```
Show me a sales dashboard for the month
```

**What happens:**
- Agent calls `show_dashboard` tool
- Server returns complete dashboard with multiple metrics
- Interactive dashboard opens in browser

### Example 3: External Link
```
Give me access to the external dashboard for project-alpha
```

**What happens:**
- Agent calls `external_dashboard` tool
- Server returns URL via `text/uri-list`
- Link displayed and opened in browser

## What You'll See

### In the Console

```
╭─ MCP-UI Resources ────────────────────────────╮
│                                                │
│  📊 chart:monthly-sales                       │
│     file://.../chart_monthly-sales.html       │
│                                                │
╰────────────────────────────────────────────────╯
```

### In Your Browser

Beautiful, interactive visualizations with:
- Responsive design
- Interactive charts (Chart.js)
- Modern UI styling
- Sandboxed for security

### In Your File System

```
.fast-agent/
└── ui/
    ├── chart_monthly-sales.html
    ├── dashboard_sales-month.html
    └── external_project-alpha.html
```

## Customization

### Change UI Mode

Edit `fastagent.config.yaml`:

```yaml
# Disable auto-open (just show links)
mcp_ui_mode: enabled

# Disable UI processing entirely
mcp_ui_mode: disabled
```

Or change it at runtime in the agent:

```python
agent.set_ui_mode("enabled")  # Disable auto-open
```

### Change Output Directory

Edit `fastagent.config.yaml`:

```yaml
# Save UI files to a custom directory
mcp_ui_output_dir: "./my-ui-files"

# Or use an absolute path
mcp_ui_output_dir: "/tmp/fast-agent-ui"
```

### Use Your Own MCP Server

Replace the mock server in `fastagent.config.yaml`:

```yaml
servers:
  - name: my-visualization-server
    command: node
    args:
      - path/to/your/server.js
```

Make sure your server returns UI resources:

```javascript
// In your MCP server
{
  content: [
    {
      type: "text",
      text: "Here's the visualization"
    },
    {
      type: "resource",
      resource: {
        uri: "ui://my-viz/chart-123",
        mimeType: "text/html",
        text: "<html>...</html>"
      }
    }
  ]
}
```

## Troubleshooting

### Browser Doesn't Open

**Solution 1:** Check your UI mode:
```bash
# In the agent, check current mode
agent._ui_mode  # Should be "auto"
```

**Solution 2:** Manually open the files:
```bash
# Open the generated HTML files
open .fast-agent/ui/*.html  # macOS
start .fast-agent/ui/*.html  # Windows
xdg-open .fast-agent/ui/*.html  # Linux
```

**Solution 3:** Use enabled mode and click links:
```yaml
mcp_ui_mode: enabled  # Links will be clickable in terminal
```

### No UI Resources Appearing

**Check 1:** Verify your server returns `ui://` URIs:
```bash
# Test your MCP server directly
# Check that tool results include ui:// resources
```

**Check 2:** Check agent configuration:
```python
print(agent._ui_mode)  # Should not be "disabled"
```

**Check 3:** Look at message history:
```python
# UI resources stored in previous user message
if len(agent.message_history) >= 2:
    msg = agent.message_history[-2]
    ui_res = msg.channels.get('mcp-ui', [])
    print(f"Found {len(ui_res)} UI resources")
```

### Permission Errors

**Issue:** Can't write to `.fast-agent/ui/`

**Solution:** Check write permissions:
```bash
# Ensure current directory is writable
ls -la .

# Or change output directory to a writable location
# Edit fastagent.config.yaml:
mcp_ui_output_dir: "$HOME/.fast-agent-ui"
```

### Security Warnings in Browser

**Issue:** Browser warns about local HTML files

**Explanation:** This is normal for `file://` URLs. The content is sandboxed.

**Solution (optional):** Serve via HTTP:
```python
# Add to your agent script
import http.server
import threading
from pathlib import Path

def serve_ui(port=8000):
    handler = http.server.SimpleHTTPRequestHandler
    httpd = http.server.HTTPServer(("", port), handler)
    httpd.serve_forever()

# Run in background
threading.Thread(target=serve_ui, daemon=True).start()
print("UI server running at http://localhost:8000/.fast-agent/ui/")
```

## Next Steps

1. **Read the Full Documentation:** See `README.md` for comprehensive details

2. **Explore the Code:**
   - `mock_ui_server.py` - Learn how to build UI-capable MCP servers
   - `simple_ui_agent.py` - See how to integrate UI support in agents

3. **Build Your Own:**
   - Create custom MCP servers that return visualizations
   - Integrate with existing dashboards
   - Build domain-specific UI tools

4. **Advanced Topics:**
   - Custom UI processing
   - FastAPI integration
   - Real-time updates
   - State persistence

## Example Projects to Try

1. **Data Analysis Dashboard**
   - Connect to pandas/polars
   - Generate charts from data
   - Interactive filtering

2. **Log Visualizer**
   - Parse log files
   - Create timeline visualizations
   - Error analysis dashboards

3. **System Monitor**
   - Real-time system metrics
   - Resource usage graphs
   - Alert dashboards

4. **Document Viewer**
   - Render Markdown/PDF
   - Syntax highlighted code
   - Interactive previews

## Resources

### In This Repository
- Full Developer Guide: `README.md` (in this directory)
- Mock Server Example: `mock_ui_server.py`
- Agent Example: `simple_ui_agent.py`
- Configuration Example: `fastagent.config.yaml`

### Official Documentation
- fast-agent MCP-UI docs: https://fast-agent.ai/mcp/mcp-ui/
- fast-agent main docs: https://fast-agent.ai
- CLI Usage & Setup: https://fast-agent.ai/mcp/mcp-ui/

### Source Code & Specs
- Source Code: `src/fast_agent/mcp/ui_agent.py`
- Tests: `tests/integration/mcp_ui/`
- MCP Protocol: https://modelcontextprotocol.io/

## Support

Issues or questions?
- GitHub Issues: https://github.com/evalstate/fast-agent/issues
- Discord: https://discord.gg/xg5cJ7ndN6

---

Happy building! 🚀
