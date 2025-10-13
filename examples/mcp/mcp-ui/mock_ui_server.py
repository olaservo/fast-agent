"""
Mock MCP Server with UI Support

This example demonstrates how to build an MCP server that returns
UI resources (visualizations, dashboards, etc.) alongside tool responses.

This server provides sample tools that return various types of UI content:
- HTML visualizations
- External dashboard URLs
- Interactive charts

Usage:
    # In your fastagent.config.yaml, add:
    servers:
      - name: mock-ui-server
        command: uv
        args:
          - run
          - mock_ui_server.py
"""

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    TextContent,
    TextResourceContents,
    Tool,
)
from pydantic import AnyUrl


# Create the MCP server
app = Server("mock-ui-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="create_chart",
            description="Create a simple bar chart visualization",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Chart title",
                    },
                    "data": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Data points for the chart",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels for each data point",
                    },
                },
                "required": ["title", "data", "labels"],
            },
        ),
        Tool(
            name="show_dashboard",
            description="Display an analytics dashboard with multiple visualizations",
            inputSchema={
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "Metric to display (sales, users, performance)",
                        "enum": ["sales", "users", "performance"],
                    },
                    "period": {
                        "type": "string",
                        "description": "Time period",
                        "enum": ["day", "week", "month", "year"],
                    },
                },
                "required": ["metric", "period"],
            },
        ),
        Tool(
            name="external_dashboard",
            description="Link to an external hosted dashboard",
            inputSchema={
                "type": "object",
                "properties": {
                    "dashboard_id": {
                        "type": "string",
                        "description": "Dashboard identifier",
                    }
                },
                "required": ["dashboard_id"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> CallToolResult:
    """Handle tool calls."""
    if name == "create_chart":
        return await create_chart(arguments)
    elif name == "show_dashboard":
        return await show_dashboard(arguments)
    elif name == "external_dashboard":
        return await external_dashboard(arguments)
    else:
        return CallToolResult(
            content=[TextContent(type="text", text=f"Unknown tool: {name}")],
            isError=True,
        )


async def create_chart(arguments: dict) -> CallToolResult:
    """Create a bar chart visualization."""
    title = arguments.get("title", "Chart")
    data = arguments.get("data", [])
    labels = arguments.get("labels", [])

    # Generate HTML with Chart.js
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .container {{
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 30px;
            max-width: 800px;
            width: 100%;
        }}
        h1 {{
            color: #333;
            margin: 0 0 20px 0;
            font-size: 24px;
        }}
        .chart-wrapper {{
            position: relative;
            height: 400px;
        }}
        .footer {{
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #666;
            font-size: 12px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="chart-wrapper">
            <canvas id="myChart"></canvas>
        </div>
        <div class="footer">
            Generated by mock-ui-server via MCP-UI
        </div>
    </div>
    <script>
        const ctx = document.getElementById('myChart');
        new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    label: 'Value',
                    data: {json.dumps(data)},
                    backgroundColor: 'rgba(102, 126, 234, 0.8)',
                    borderColor: 'rgba(102, 126, 234, 1)',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    # Create UI resource
    ui_resource = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=AnyUrl(f"ui://chart/{title.lower().replace(' ', '-')}"),
            mimeType="text/html",
            text=html_content,
        ),
    )

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Created a bar chart titled '{title}' with {len(data)} data points. The visualization is available below.",
            ),
            ui_resource,
        ]
    )


async def show_dashboard(arguments: dict) -> CallToolResult:
    """Show an analytics dashboard."""
    metric = arguments.get("metric", "sales")
    period = arguments.get("period", "month")

    # Generate mock data
    import random

    random.seed(hash(metric + period))
    data_points = [random.randint(50, 150) for _ in range(12)]
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{metric.title()} Dashboard - {period.title()}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            font-size: 28px;
            margin-bottom: 5px;
        }}
        .header p {{
            opacity: 0.9;
            font-size: 14px;
        }}
        .dashboard {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .stat-label {{
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }}
        .stat-change {{
            font-size: 14px;
            color: #10b981;
            margin-top: 5px;
        }}
        .chart-container {{
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .chart-wrapper {{
            position: relative;
            height: 400px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{metric.title()} Dashboard</h1>
        <p>Period: {period.title()} | Last updated: just now</p>
    </div>
    <div class="dashboard">
        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">Total {metric.title()}</div>
                <div class="stat-value">{sum(data_points):,}</div>
                <div class="stat-change">↑ 12% from last {period}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Average</div>
                <div class="stat-value">{sum(data_points)//len(data_points)}</div>
                <div class="stat-change">↑ 8% from last {period}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Peak</div>
                <div class="stat-value">{max(data_points)}</div>
                <div class="stat-change">↑ 15% from last {period}</div>
            </div>
        </div>
        <div class="chart-container">
            <h2 style="margin-bottom: 20px; color: #333;">Trend Analysis</h2>
            <div class="chart-wrapper">
                <canvas id="trendChart"></canvas>
            </div>
        </div>
    </div>
    <script>
        const ctx = document.getElementById('trendChart');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(months)},
                datasets: [{{
                    label: '{metric.title()}',
                    data: {json.dumps(data_points)},
                    borderColor: 'rgba(102, 126, 234, 1)',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    tension: 0.4,
                    fill: true
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    ui_resource = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=AnyUrl(f"ui://dashboard/{metric}-{period}"),
            mimeType="text/html",
            text=html_content,
        ),
    )

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Here's the {metric} dashboard for the {period} period. The interactive visualization shows trends and key metrics.",
            ),
            ui_resource,
        ]
    )


async def external_dashboard(arguments: dict) -> CallToolResult:
    """Link to an external dashboard (demonstrates text/uri-list)."""
    dashboard_id = arguments.get("dashboard_id", "demo")

    # Example: Link to a public dashboard
    # In production, this would be your actual dashboard URL
    dashboard_url = f"https://example.com/dashboard/{dashboard_id}"

    ui_resource = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=AnyUrl(f"ui://external/{dashboard_id}"),
            mimeType="text/uri-list",
            text=f"{dashboard_url}\n",
        ),
    )

    return CallToolResult(
        content=[
            TextContent(
                type="text",
                text=f"Your dashboard is ready. Click the link below to view it in a new window.",
            ),
            ui_resource,
        ]
    )


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
