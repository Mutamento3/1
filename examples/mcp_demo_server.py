#!/usr/bin/env python3
"""
mcp_demo_server.py — a tiny stdio MCP server to prove the api_loop tool path end to end.

Register it in the PWA settings page (MCP 工具 → 添加):
    name: demo   transport: stdio   command: python3   args: mcp_demo_server.py

Then ask the model "现在几点" / "把 3.5 和 4 加起来" / "写一张便签叫 hello 内容 xxx 然后发给我".
The last one exercises write_note → attach_file: the note comes back as a real attachment.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo")
NOTES_DIR = Path(__file__).resolve().parent / "loop_cache" / "notes"


@mcp.tool()
def now(tz: str = "UTC") -> str:
    """Current date and time in the given IANA timezone (e.g. Asia/Shanghai)."""
    try:
        zone = ZoneInfo(tz)
    except Exception:
        zone = ZoneInfo("UTC")
    return dt.datetime.now(zone).strftime("%Y-%m-%d %H:%M:%S %Z")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool()
def write_note(name: str, text: str) -> str:
    """Save a text note on the server and return its path (send it to the user with attach_file)."""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c for c in name if c.isalnum() or c in "-_.") or "note"
    path = NOTES_DIR / (safe if safe.endswith(".txt") else safe + ".txt")
    path.write_text(text, encoding="utf-8")
    return str(path)


if __name__ == "__main__":
    import sys
    # `python3 mcp_demo_server.py --http 8765` serves the same tools over streamable-http at
    # http://127.0.0.1:8765/mcp — register it with transport=http to test that path.
    if len(sys.argv) >= 3 and sys.argv[1] == "--http":
        mcp.settings.host, mcp.settings.port = "127.0.0.1", int(sys.argv[2])
        mcp.run(transport="streamable-http")
    else:
        mcp.run()  # stdio
