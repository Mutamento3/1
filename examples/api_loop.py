#!/usr/bin/env python3
"""
api_loop.py — server-side API body for Tidal Echo.

Runs beside backend/app.py. When the relay's /app/brain is "loop", every human
message is POSTed here (/loop/ingest); this file turns it into a full model turn
and posts the answer back through /channel/out.

One turn, in order:
  1. persona      system prompt read from a file; the file is re-read whenever
                  its mtime changes (edit it in the PWA settings page or with any
                  editor — no restart).
  2. context      the last history_n messages of the same PWA session, straight
                  from relay.db.
  3. attachments  images the human sent become image_url parts (multimodal),
                  small text-like files are inlined, everything else becomes a
                  one-line note + a local copy the tools can read.
  4. tools        MCP servers (stdio or streamable-http) are connected at start;
                  their tools are offered to the model as function calls and
                  tool_calls run in a loop (max_tool_steps). Every step is shown
                  in the PWA as an expandable "act" chip.
  5. reply        streamed to the PWA as reply_delta and finalised as one reply.
                  Files the model attaches (attach_file tool) and images returned
                  by MCP tools ride along as attachments.

All private values live in .env or api_loop.config.json (both git-ignored).
Nothing here is tied to a person, a domain or a model vendor.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import datetime as dt
import json
import mimetypes
import os
import re
import sqlite3
import uuid
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request

try:  # MCP is optional: without the SDK the loop still chats, just without tools.
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    MCP_AVAILABLE = True
except Exception:  # pragma: no cover - import guard
    MCP_AVAILABLE = False


def load_dotenv(path: Path) -> None:
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

LOOP_PORT = int(os.environ.get("LOOP_PORT", "3020"))
LOOP_CONFIG = Path(os.environ.get("LOOP_CONFIG", str(HERE / "api_loop.config.json")))
LOOP_CACHE_DIR = Path(os.environ.get("LOOP_CACHE_DIR", str(HERE / "loop_cache")))
RELAY_DB = os.environ.get("RELAY_DB", str(HERE.parent / "backend" / "relay.db"))
RELAY_UPLOAD_DIR = os.environ.get("RELAY_UPLOAD_DIR", "")
RELAY_URL = os.environ.get("RELAY_URL", "http://127.0.0.1:3011").rstrip("/")
RELAY_SECRET = os.environ.get("RELAY_SECRET", "")
PERSONA_FILE_ENV = os.environ.get("PERSONA_FILE", "").strip()
PERSONA_ENV = os.environ.get("PERSONA", "").strip()
HISTORY_N = int(os.environ.get("HISTORY_N", "24"))
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "2000"))
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
STREAM_OUTPUT = os.environ.get("LOOP_STREAM", "1").lower() not in {"0", "false", "no"}
FALLBACK_CODES = {401, 403, 404, 408, 409, 429, 500, 502, 503, 504}
DEFAULT_PERSONA = (
    "You are the user's private AI companion in a one-to-one chat. "
    "Reply naturally, warmly, and concisely unless the user asks for detail."
)

IMAGE_MAX_BYTES = 8 * 1024 * 1024
TEXT_INLINE_MAX = 64 * 1024
TEXTISH_EXT = {
    ".txt", ".md", ".markdown", ".json", ".csv", ".tsv", ".py", ".js", ".ts", ".html", ".css",
    ".yaml", ".yml", ".toml", ".ini", ".log", ".xml", ".sh", ".sql", ".srt", ".vtt",
}
# attach_file never sends files whose name looks like a secret, whatever the path:
# .env / *.env / .env.* , *.pem, *.key, relay.db*, api_loop.config.json, id_rsa*, *.p12/*.pfx,
# and anything with "secret", "credential" or "token" in the name.
SECRET_NAME_RE = re.compile(
    r"((^|\.)env(\.|$)|\.pem$|\.key$|relay\.db|api_loop\.config\.json|id_rsa|\.p12$|\.pfx$|secret|credential|token)", re.I
)

CONFIG_DEFAULTS: dict[str, Any] = {
    "history_n": HISTORY_N,     # how many earlier messages of this session go to the model
    "history_images": 2,        # of those, how many recent human images are re-sent as pixels
    "max_tool_steps": 8,        # tool_calls rounds per turn before the model must answer
    "vision": "auto",           # auto | on | off — send images as image_url parts
    "persona_file": "",         # path; empty = PERSONA_FILE env, then PERSONA env, then default
    "attach_roots": [],         # optional allow-list of directories attach_file may read from
    "mcp_servers": [],          # [{name, transport: stdio|http, command, args, env, url, headers, enabled}]
}


# ---------------------------------------------------------------------------
# config file
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def mask_key(key: str) -> str:
    key = str(key or "")
    if not key:
        return ""
    if len(key) <= 10:
        return "***"
    return key[:6] + "***" + key[-4:]


def load_config() -> dict[str, Any]:
    try:
        data = json.loads(LOOP_CONFIG.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def save_config(cfg: dict[str, Any]) -> None:
    LOOP_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    tmp = LOOP_CONFIG.with_suffix(LOOP_CONFIG.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(LOOP_CONFIG)
    with contextlib.suppress(OSError):
        os.chmod(LOOP_CONFIG, 0o600)  # it holds API keys


def cfg_int(name: str, lo: int, hi: int) -> int:
    try:
        return max(lo, min(int(load_config().get(name, CONFIG_DEFAULTS[name])), hi))
    except Exception:
        return int(CONFIG_DEFAULTS[name])


def env_routes() -> list[dict[str, str]]:
    routes: list[dict[str, str]] = []
    for suffix in ("", "_2", "_3", "_4"):
        base = os.environ.get(f"LLM_API_BASE{suffix}", "").rstrip("/")
        key = os.environ.get(f"LLM_API_KEY{suffix}", "")
        model = os.environ.get(f"LLM_MODEL{suffix}", "")
        if base and key and model:
            routes.append({"url": base, "key": key, "model": model})
    return routes


def main_chain() -> list[dict[str, str]]:
    cfg = load_config()
    configured = cfg.get("main_chain")
    if isinstance(configured, list):
        rows = [r for r in configured if isinstance(r, dict) and r.get("url") and r.get("key") and r.get("model")]
        if rows:
            return rows
    return env_routes()


def history_n() -> int:
    return cfg_int("history_n", 0, 200)


# ---------------------------------------------------------------------------
# persona — a file, re-read on change
# ---------------------------------------------------------------------------

_persona_cache: dict[str, Any] = {"path": "", "mtime": None, "text": ""}


def persona_path() -> Path | None:
    raw = str(load_config().get("persona_file") or PERSONA_FILE_ENV or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (HERE / path)


def persona_text() -> str:
    path = persona_path()
    if path is not None:
        try:
            st = path.stat()
            if _persona_cache["path"] != str(path) or _persona_cache["mtime"] != st.st_mtime:
                _persona_cache.update(path=str(path), mtime=st.st_mtime, text=path.read_text(encoding="utf-8").strip())
            if _persona_cache["text"]:
                return _persona_cache["text"]
        except OSError:
            pass
    return PERSONA_ENV or DEFAULT_PERSONA


def persona_public() -> dict[str, Any]:
    path = persona_path()
    text = persona_text()
    source = "default"
    mtime = ""
    exists = False
    if path is not None and path.exists():
        source, exists = "file", True
        mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat()
    elif PERSONA_ENV:
        source = "env"
    return {
        "text": text,
        "path": str(path) if path is not None else "",
        "exists": exists,
        "mtime": mtime,
        "chars": len(text),
        "source": source,
    }


def persona_save(text: str) -> dict[str, Any]:
    path = persona_path() or (HERE / "persona.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text.strip() + "\n", encoding="utf-8")
    tmp.replace(path)
    cfg = load_config()
    if not str(cfg.get("persona_file") or "").strip():
        cfg["persona_file"] = str(path)
        save_config(cfg)
    _persona_cache["mtime"] = None  # force re-read on next turn
    return persona_public()


# ---------------------------------------------------------------------------
# sessions (PWA multi-window)
# ---------------------------------------------------------------------------

def session_rows() -> list[dict[str, Any]]:
    rows = load_config().get("sessions")
    if not isinstance(rows, list):
        return []
    out = []
    for item in rows:
        if isinstance(item, dict) and item.get("id"):
            out.append({
                "id": str(item.get("id")),
                "title": str(item.get("title") or "New chat"),
                "since_id": int(item.get("since_id") or 0),
                "created_at": item.get("created_at") or "",
                "pinned": bool(item.get("pinned", False)),
            })
    return out


def active_session_id() -> str:
    cfg = load_config()
    active = str(cfg.get("active_session") or "").strip()
    ids = {s["id"] for s in session_rows()}
    if active in ids:
        return active
    rows = session_rows()
    return rows[-1]["id"] if rows else ""


def save_sessions(rows: list[dict[str, Any]], active: str | None = None) -> dict[str, Any]:
    cfg = load_config()
    cfg["sessions"] = rows
    if active is not None:
        cfg["active_session"] = active
    save_config(cfg)
    return sessions_public()


def sessions_public() -> dict[str, Any]:
    return {"active_session": active_session_id(), "sessions": session_rows()}


def create_session(title: str = "New chat", since_id: int = 0, activate: bool = True) -> dict[str, Any]:
    rows = session_rows()
    sid = "api-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    row = {"id": sid, "title": title or "New chat", "since_id": int(since_id or 0), "created_at": now_iso()}
    rows.append(row)
    save_sessions(rows, sid if activate else None)
    return row


def patch_session(session_id: str, body: dict[str, Any]) -> dict[str, Any]:
    rows = session_rows()
    found = False
    for item in rows:
        if item["id"] != session_id:
            continue
        found = True
        if "title" in body:
            item["title"] = str(body.get("title") or item["title"]).strip() or item["title"]
        if "pinned" in body:
            item["pinned"] = bool(body.get("pinned"))
    if not found:
        raise HTTPException(status_code=404, detail="session not found")
    active = session_id if body.get("active") else None
    return save_sessions(rows, active)


# ---------------------------------------------------------------------------
# relay I/O
# ---------------------------------------------------------------------------

def relay_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    h = {"Authorization": f"Bearer {RELAY_SECRET}"}
    if extra:
        h.update(extra)
    return h


async def relay_out(payload: dict[str, Any]) -> tuple[bool, Any]:
    if not RELAY_SECRET:
        return False, "RELAY_SECRET missing"
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        resp = await client.post(
            f"{RELAY_URL}/channel/out",
            headers=relay_headers({"Content-Type": "application/json"}),
            json=payload,
        )
    try:
        body: Any = resp.json()
    except Exception:
        body = resp.text[:500]
    return resp.status_code < 300, body


async def relay_upload(data: bytes, name: str, mime: str) -> dict[str, Any] | None:
    """Push bytes into the relay's upload store; returns the attachment dict the PWA renders."""
    if not RELAY_SECRET:
        return None
    async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
        resp = await client.post(
            f"{RELAY_URL}/app/upload",
            params={"name": name},
            headers=relay_headers({"Content-Type": mime or "application/octet-stream"}),
            content=data,
        )
    if resp.status_code >= 300:
        return None
    try:
        out = resp.json()
        return out if isinstance(out, dict) and out.get("url") else None
    except Exception:
        return None


async def fetch_upload(att: dict[str, Any]) -> tuple[bytes | None, Path | None]:
    """Get the bytes of an attachment the human sent. Local upload dir first, relay HTTP second.
    A copy is kept under LOOP_CACHE_DIR so tools (e.g. a filesystem MCP) can open it."""
    stored = Path(str(att.get("url") or "")).name
    if not stored:
        return None, None
    data: bytes | None = None
    if RELAY_UPLOAD_DIR:
        local = Path(RELAY_UPLOAD_DIR) / stored
        if local.is_file():
            with contextlib.suppress(OSError):
                data = local.read_bytes()
    if data is None and RELAY_SECRET:
        try:
            async with httpx.AsyncClient(timeout=60, trust_env=False) as client:
                resp = await client.get(f"{RELAY_URL}/uploads/{stored}", headers=relay_headers())
            if resp.status_code < 300:
                data = resp.content
        except Exception:
            data = None
    if data is None:
        return None, None
    cache = LOOP_CACHE_DIR / "in" / stored
    with contextlib.suppress(OSError):
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
    return data, cache


# ---------------------------------------------------------------------------
# attachments → message parts
# ---------------------------------------------------------------------------

def vision_enabled() -> bool:
    return str(load_config().get("vision", CONFIG_DEFAULTS["vision"])).lower() != "off"


def is_textish(name: str, mime: str) -> bool:
    return (mime or "").startswith("text/") or Path(name or "").suffix.lower() in TEXTISH_EXT


def att_label(att: dict[str, Any]) -> str:
    kind = att.get("kind") or ("image" if str(att.get("mime") or "").startswith("image/") else "file")
    return f"[{'图片' if kind == 'image' else '文件'}: {att.get('name') or 'file'}]"


async def attachment_parts(atts: list[dict[str, Any]], *, with_images: bool) -> tuple[list[dict[str, Any]], list[str]]:
    """→ (image_url parts, text notes). Notes always exist so a text-only model still knows what arrived."""
    parts: list[dict[str, Any]] = []
    notes: list[str] = []
    for att in atts or []:
        if not isinstance(att, dict):
            continue
        name = str(att.get("name") or "file")
        mime = str(att.get("mime") or mimetypes.guess_type(name)[0] or "application/octet-stream")
        size = int(att.get("size") or 0)
        is_image = mime.startswith("image/")
        data, local = await fetch_upload(att)
        if data is None:
            notes.append(f"{att_label(att)} (无法读取附件内容)")
            continue
        if is_image and with_images and len(data) <= IMAGE_MAX_BYTES:
            parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"}})
            notes.append(att_label(att))
        elif is_textish(name, mime) and len(data) <= TEXT_INLINE_MAX:
            try:
                body = data.decode("utf-8")
            except UnicodeDecodeError:
                body = data.decode("utf-8", "replace")
            notes.append(f"[文件 {name}]\n```\n{body.strip()}\n```")
        else:
            where = f" · 已存到 {local}" if local else ""
            notes.append(f"[附件: {name} · {mime} · {size or len(data)} B{where}]")
    return parts, notes


def user_content(text: str, notes: list[str], parts: list[dict[str, Any]]) -> Any:
    body = "\n".join([t for t in [text.strip()] + notes if t]).strip()
    if not parts:
        return body
    content: list[dict[str, Any]] = []
    if body:
        content.append({"type": "text", "text": body})
    content.extend(parts)
    return content


# ---------------------------------------------------------------------------
# context from relay.db
# ---------------------------------------------------------------------------

def relay_rows(before_id: int | None, session_id: str, limit: int) -> list[dict[str, Any]]:
    path = Path(RELAY_DB)
    if not path.exists() or limit <= 0:
        return []
    params: list[Any] = []
    where = ["kind IN ('user','voice','reply')"]
    if before_id:
        where.append("id < ?")
        params.append(int(before_id))
    if session_id:
        where.append("json_extract(meta, '$.api_session') = ?")
        params.append(session_id)
    else:
        where.append("(json_extract(meta, '$.api_session') IS NULL OR json_extract(meta, '$.api_session') = '')")
    sql = (
        "SELECT id, direction, kind, text, meta FROM messages "
        f"WHERE {' AND '.join(where)} ORDER BY id DESC LIMIT ?"
    )
    params.append(max(0, limit))
    with sqlite3.connect(str(path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    out = []
    for r in reversed(rows):
        d = dict(r)
        try:
            d["meta"] = json.loads(d.get("meta") or "{}")
        except Exception:
            d["meta"] = {}
        out.append(d)
    return out


def relay_message_attachments(msg_id: int | None) -> list[dict[str, Any]]:
    """Fallback for relays that forward only {id,text}: read the attachments off the stored row."""
    if not msg_id:
        return []
    path = Path(RELAY_DB)
    if not path.exists():
        return []
    try:
        with sqlite3.connect(str(path)) as conn:
            row = conn.execute("SELECT meta FROM messages WHERE id = ?", (int(msg_id),)).fetchone()
        meta = json.loads(row[0] or "{}") if row else {}
        atts = meta.get("attachments")
        return atts if isinstance(atts, list) else []
    except Exception:
        return []


async def build_messages(
    text: str,
    atts: list[dict[str, Any]],
    *,
    before_id: int | None = None,
    session_id: str = "",
    use_context: bool = True,
    with_images: bool = True,
) -> tuple[list[dict[str, Any]], bool]:
    """→ (messages, had_images). History images are only re-sent for the newest history_images human rows."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": persona_text()}]
    had_images = False
    if use_context:
        rows = relay_rows(before_id, session_id, history_n())
        budget = cfg_int("history_images", 0, 20) if with_images else 0
        image_rows: set[int] = set()
        for row in reversed(rows):
            if budget <= 0:
                break
            if row.get("direction") == "in" and any(str((a or {}).get("mime") or "").startswith("image/") for a in row["meta"].get("attachments") or []):
                image_rows.add(int(row["id"]))
                budget -= 1
        for row in rows:
            content = str(row.get("text") or "").strip()
            row_atts = row["meta"].get("attachments") or []
            if row.get("direction") == "out":
                if content:
                    messages.append({"role": "assistant", "content": content})
                continue
            if int(row["id"]) in image_rows:
                parts, notes = await attachment_parts(row_atts, with_images=True)
                had_images = had_images or bool(parts)
                body = user_content(content, notes, parts)
            else:
                body = "\n".join([t for t in [content] + [att_label(a) for a in row_atts if isinstance(a, dict)] if t])
            if body:
                messages.append({"role": "user", "content": body})
    parts, notes = await attachment_parts(atts, with_images=with_images)
    had_images = had_images or bool(parts)
    messages.append({"role": "user", "content": user_content(text, notes, parts) or "(空消息)"})
    return messages, had_images


# ---------------------------------------------------------------------------
# MCP servers → tools
# ---------------------------------------------------------------------------

TOOL_NAME_RE = re.compile(r"[^A-Za-z0-9_-]+")


def tool_public_name(server: str, tool: str) -> str:
    name = TOOL_NAME_RE.sub("_", f"{server}__{tool}").strip("_")
    return name[:64] or "tool"


class McpServer:
    """One MCP server held open in its own task (anyio transports must enter/exit in one task)."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.name = str(cfg.get("name") or "mcp")
        self.session: ClientSession | None = None
        self.tools: list[Any] = []
        self.status = "offline"
        self.error = ""
        self.connected_at = ""
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._ready = asyncio.Event()

    @property
    def transport(self) -> str:
        return "http" if str(self.cfg.get("transport") or "stdio").lower() in {"http", "streamable-http", "streamable_http"} else "stdio"

    async def _run(self) -> None:
        try:
            async with AsyncExitStack() as stack:
                if self.transport == "http":
                    read, write, _ = await stack.enter_async_context(
                        streamablehttp_client(str(self.cfg.get("url") or ""), headers=dict(self.cfg.get("headers") or {}))
                    )
                else:
                    env = {**os.environ, **{str(k): str(v) for k, v in (self.cfg.get("env") or {}).items()}}
                    params = StdioServerParameters(
                        command=str(self.cfg.get("command") or ""),
                        args=[str(a) for a in (self.cfg.get("args") or [])],
                        env=env,
                        cwd=str(self.cfg.get("cwd") or HERE),
                    )
                    read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await asyncio.wait_for(session.initialize(), timeout=30)
                listed = await asyncio.wait_for(session.list_tools(), timeout=30)
                self.tools = list(listed.tools or [])
                self.session = session
                self.status, self.error, self.connected_at = "online", "", now_iso()
                self._ready.set()
                await self._stop.wait()
        except Exception as exc:
            self.status = "error"
            self.error = f"{type(exc).__name__}: {exc}"[:400]
        finally:
            self.session = None
            self.tools = []
            if self.status != "error":
                self.status = "offline"
            self._ready.set()

    async def start(self, wait: float = 20.0) -> None:
        if self._task and not self._task.done():
            return
        self._stop = asyncio.Event()
        self._ready = asyncio.Event()
        self.status, self.error = "connecting", ""
        self._task = asyncio.create_task(self._run(), name=f"mcp:{self.name}")
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._ready.wait(), timeout=wait)

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._task, timeout=10)
        self._task = None

    async def call(self, tool: str, args: dict[str, Any]) -> Any:
        if self.session is None:
            await self.start()
        if self.session is None:
            raise RuntimeError(f"MCP server '{self.name}' is {self.status}: {self.error or 'not connected'}")
        return await asyncio.wait_for(self.session.call_tool(tool, args or {}), timeout=120)

    def public(self) -> dict[str, Any]:
        cfg = dict(self.cfg)
        if isinstance(cfg.get("headers"), dict):
            cfg["headers"] = {k: mask_key(v) if "auth" in k.lower() or "key" in k.lower() or "token" in k.lower() else v for k, v in cfg["headers"].items()}
        if isinstance(cfg.get("env"), dict):
            cfg["env"] = {k: mask_key(v) if any(s in k.lower() for s in ("key", "secret", "token", "pass")) else v for k, v in cfg["env"].items()}
        return {
            **cfg,
            "name": self.name,
            "transport": self.transport,
            "status": self.status,
            "error": self.error,
            "connected_at": self.connected_at,
            "tools": [{"name": t.name, "description": (t.description or "")[:200]} for t in self.tools],
        }


class McpManager:
    def __init__(self) -> None:
        self.servers: dict[str, McpServer] = {}
        self.index: dict[str, tuple[str, str]] = {}  # public tool name → (server, tool)

    def configured(self) -> list[dict[str, Any]]:
        rows = load_config().get("mcp_servers")
        return [r for r in rows if isinstance(r, dict) and r.get("name")] if isinstance(rows, list) else []

    async def start_all(self) -> None:
        if not MCP_AVAILABLE:
            return
        for cfg in self.configured():
            if cfg.get("enabled", True):
                await self.restart(cfg["name"], cfg)

    async def stop_all(self) -> None:
        for s in list(self.servers.values()):
            await s.stop()
        self.servers.clear()
        self.index.clear()

    async def restart(self, name: str, cfg: dict[str, Any] | None = None) -> McpServer | None:
        if not MCP_AVAILABLE:
            raise HTTPException(status_code=501, detail="pip install mcp  (MCP SDK not installed)")
        old = self.servers.pop(name, None)
        if old:
            await old.stop()
        cfg = cfg or next((c for c in self.configured() if c.get("name") == name), None)
        if not cfg:
            self.rebuild_index()
            return None
        server = McpServer(cfg)
        self.servers[name] = server
        if cfg.get("enabled", True):
            await server.start()
        self.rebuild_index()
        return server

    def rebuild_index(self) -> None:
        self.index = {}
        for s in self.servers.values():
            for t in s.tools:
                self.index[tool_public_name(s.name, t.name)] = (s.name, t.name)

    def openai_tools(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for s in self.servers.values():
            for t in s.tools:
                schema = t.inputSchema if isinstance(getattr(t, "inputSchema", None), dict) else {}
                if schema.get("type") != "object":
                    schema = {"type": "object", "properties": schema.get("properties") or {}}
                out.append({
                    "type": "function",
                    "function": {
                        "name": tool_public_name(s.name, t.name),
                        "description": (t.description or f"{t.name} on MCP server {s.name}")[:1024],
                        "parameters": schema,
                    },
                })
        return out

    def public(self) -> list[dict[str, Any]]:
        rows = []
        for cfg in self.configured():
            s = self.servers.get(cfg["name"])
            if s:
                rows.append(s.public())
            else:
                rows.append({**cfg, "status": "disabled" if not cfg.get("enabled", True) else "offline", "error": "", "tools": []})
        return rows


mcp_manager = McpManager()

BUILTIN_TOOLS: list[dict[str, Any]] = [{
    "type": "function",
    "function": {
        "name": "attach_file",
        "description": "Send a file that exists on this server to the user as an attachment of your current reply "
                       "(images show inline). Use it for files you or a tool just produced.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path, or relative to the loop directory"},
                "name": {"type": "string", "description": "Optional display name"},
            },
            "required": ["path"],
        },
    },
}]


def all_tools() -> list[dict[str, Any]]:
    return BUILTIN_TOOLS + mcp_manager.openai_tools()


def attach_allowed(path: Path) -> bool:
    if SECRET_NAME_RE.search(path.name):
        return False
    roots = [r for r in (load_config().get("attach_roots") or []) if isinstance(r, str) and r.strip()]
    if not roots:
        return True
    for r in roots:
        root = Path(r).expanduser()
        root = root if root.is_absolute() else HERE / root
        with contextlib.suppress(ValueError):
            path.resolve().relative_to(root.resolve())
            return True
    return False


class Turn:
    """Per-message scratch: attachments queued by tools, act chips posted, usage."""

    def __init__(self, session_id: str, *, dry: bool):
        self.session_id = session_id
        self.dry = dry
        self.attachments: list[dict[str, Any]] = []
        self.acts: list[dict[str, Any]] = []
        self.usage: dict[str, Any] = {}

    async def attach_bytes(self, data: bytes, name: str, mime: str) -> str:
        up = await relay_upload(data, name, mime)
        if not up:
            return f"ERROR: upload of {name} failed"
        self.attachments.append(up)
        return f"attached {name} ({len(data)} B)"

    async def run_tool(self, public_name: str, raw_args: str) -> str:
        try:
            args = json.loads(raw_args) if raw_args and raw_args.strip() else {}
            if not isinstance(args, dict):
                args = {}
        except json.JSONDecodeError as exc:
            return f"ERROR: arguments are not valid JSON ({exc})"
        if public_name == "attach_file":
            raw = str(args.get("path") or "").strip()
            if not raw:
                return "ERROR: path required"
            path = Path(raw).expanduser()
            path = path if path.is_absolute() else HERE / path
            if not path.is_file():
                return f"ERROR: not a file: {path}"
            if not attach_allowed(path):
                return f"ERROR: {path.name} is not allowed to be attached"
            data = path.read_bytes()
            if len(data) > 10 * 1024 * 1024:
                return "ERROR: file larger than 10 MB"
            name = str(args.get("name") or path.name)
            mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
            return await self.attach_bytes(data, name, mime)
        target = mcp_manager.index.get(public_name)
        if not target:
            return f"ERROR: unknown tool {public_name}"
        server_name, tool = target
        server = mcp_manager.servers.get(server_name)
        if not server:
            return f"ERROR: MCP server {server_name} is not running"
        try:
            result = await server.call(tool, args)
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"[:2000]
        texts: list[str] = []
        for item in getattr(result, "content", None) or []:
            ctype = getattr(item, "type", "")
            if ctype == "text":
                texts.append(str(getattr(item, "text", "") or ""))
            elif ctype == "image":
                try:
                    data = base64.b64decode(getattr(item, "data", "") or "")
                    mime = str(getattr(item, "mimeType", "") or "image/png")
                    ext = mimetypes.guess_extension(mime) or ".png"
                    texts.append("[image] " + await self.attach_bytes(data, f"{tool}-{uuid.uuid4().hex[:6]}{ext}", mime))
                except Exception as exc:
                    texts.append(f"[image could not be attached: {exc}]")
            elif ctype == "resource":
                res = getattr(item, "resource", None)
                text = getattr(res, "text", None)
                if text:
                    texts.append(str(text))
                else:
                    texts.append(f"[resource {getattr(res, 'uri', '')}]")
        structured = getattr(result, "structuredContent", None)
        if not texts and structured:
            texts.append(json.dumps(structured, ensure_ascii=False))
        out = "\n".join(t for t in texts if t).strip() or "(empty result)"
        if getattr(result, "isError", False):
            out = "ERROR: " + out
        return out[:20000]


def act_glyph(tool: str) -> str:
    t = tool.lower()
    if any(k in t for k in ("search", "grep", "find", "query")):
        return "search"
    if any(k in t for k in ("fetch", "web", "http", "url", "browse")):
        return "web"
    if any(k in t for k in ("memory", "recall", "remember", "note")):
        return "memory"
    return "terminal"


# ---------------------------------------------------------------------------
# model calls (OpenAI-compatible, streaming with tool_calls)
# ---------------------------------------------------------------------------

class ModelError(Exception):
    def __init__(self, status: int, detail: str, route: dict[str, str] | None = None):
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.route = route or {}


def _merge_tool_call(acc: dict[int, dict[str, Any]], tc: dict[str, Any], pos: int) -> None:
    idx = tc.get("index")
    idx = int(idx) if isinstance(idx, int) else pos
    slot = acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
    if tc.get("id"):
        slot["id"] = str(tc["id"])
    fn = tc.get("function") or {}
    if fn.get("name"):
        slot["name"] = str(fn["name"])
    if fn.get("arguments"):
        slot["arguments"] += str(fn["arguments"])


def _finish_tool_calls(acc: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    calls = []
    for idx in sorted(acc):
        c = acc[idx]
        if not c.get("name"):
            continue
        calls.append({"id": c.get("id") or f"call_{uuid.uuid4().hex[:12]}", "name": c["name"], "arguments": c.get("arguments") or "{}"})
    return calls


async def _raise_for(resp: httpx.Response, route: dict[str, str]) -> None:
    if resp.status_code < 300:
        return
    try:
        body = (await resp.aread()).decode("utf-8", "replace")
    except Exception:
        body = ""
    raise ModelError(resp.status_code, body[:400] or resp.reason_phrase, route)


async def stream_chat(route: dict[str, str], messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, sink, think_sink) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": route["model"],
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }
    if tools:
        body["tools"] = tools
    text_parts: list[str] = []
    think_parts: list[str] = []
    usage: dict[str, Any] = {}
    acc: dict[int, dict[str, Any]] = {}
    # trust_env=True: model endpoints are on the internet, so HTTP(S)_PROXY / NO_PROXY apply.
    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=30), trust_env=True) as client:
        async with client.stream(
            "POST",
            route["url"].rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {route['key']}", "Content-Type": "application/json"},
            json=body,
        ) as resp:
            await _raise_for(resp, route)
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(ev.get("usage"), dict):
                    usage = ev["usage"]
                choice = (ev.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                think = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if think:
                    think_parts.append(think)
                    await think_sink(think)
                chunk = delta.get("content") or ""
                if chunk:
                    text_parts.append(chunk)
                    await sink(chunk)
                for pos, tc in enumerate(delta.get("tool_calls") or []):
                    if isinstance(tc, dict):
                        _merge_tool_call(acc, tc, pos)
    return {"text": "".join(text_parts).strip(), "thinking": "".join(think_parts).strip(), "tool_calls": _finish_tool_calls(acc), "usage": usage}


async def complete_chat(route: dict[str, str], messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": route["model"],
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
    async with httpx.AsyncClient(timeout=300, trust_env=True) as client:
        resp = await client.post(
            route["url"].rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {route['key']}", "Content-Type": "application/json"},
            json=body,
        )
    await _raise_for(resp, route)
    data = resp.json()
    msg = ((data.get("choices") or [{}])[0]).get("message") or {}
    acc: dict[int, dict[str, Any]] = {}
    for pos, tc in enumerate(msg.get("tool_calls") or []):
        if isinstance(tc, dict):
            _merge_tool_call(acc, tc, pos)
    return {
        "text": (msg.get("content") or "").strip(),
        "thinking": (msg.get("reasoning_content") or "").strip(),
        "tool_calls": _finish_tool_calls(acc),
        "usage": data.get("usage") or {},
    }


async def run_model(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, *, sink=None, think_sink=None, stream: bool = True) -> dict[str, Any]:
    """Walk the model chain. Falls back to the next route only if nothing was streamed yet."""
    tried: list[str] = []
    errors: list[str] = []
    for route in main_chain():
        tried.append(route.get("model", ""))
        started = {"v": False}

        async def _sink(chunk: str) -> None:
            started["v"] = True
            if sink:
                await sink(chunk)

        async def _think(chunk: str) -> None:
            started["v"] = True
            if think_sink:
                await think_sink(chunk)

        try:
            if stream and STREAM_OUTPUT:
                out = await stream_chat(route, messages, tools, _sink, _think)
            else:
                out = await complete_chat(route, messages, tools)
            out["model"] = route.get("model")
            out["tried"] = tried[:-1]
            return out
        except ModelError as exc:
            err = f"{route.get('model')} @ {route.get('url')}: HTTP {exc.status} {exc.detail[:160]}"
            errors.append(err)
            if started["v"] or exc.status not in FALLBACK_CODES:
                raise ModelError(exc.status, "; ".join(errors), route) from exc
        except Exception as exc:
            err = f"{route.get('model')} @ {route.get('url')}: {type(exc).__name__}: {exc}"
            errors.append(err)
            if started["v"]:
                raise ModelError(0, "; ".join(errors), route) from exc
    raise ModelError(0, "; ".join(errors) or "no model configured (fill main_chain in settings or LLM_* in .env)")


# ---------------------------------------------------------------------------
# one full turn
# ---------------------------------------------------------------------------

def _step_label(calls: list[dict[str, Any]]) -> str:
    names = [mcp_manager.index.get(c["name"], ("", c["name"]))[1] for c in calls]
    if len(names) == 1:
        return f"用了 {names[0]}"
    return f"跑了 {len(names)} 个工具"


async def handle_turn(
    text: str,
    atts: list[dict[str, Any]],
    msg_id: int | None,
    session_id: str,
    *,
    dry: bool = False,
    use_context: bool = True,
) -> dict[str, Any]:
    stream_id = "api-" + uuid.uuid4().hex[:16]
    turn = Turn(session_id, dry=dry)
    emit = (not dry) and STREAM_OUTPUT

    async def sink(chunk: str) -> None:
        if emit:
            await relay_out({"type": "reply_delta", "stream_id": stream_id, "text": chunk, "done": False, "api_session": session_id})

    think_state = {"id": "", "n": 0}

    async def think_sink(chunk: str) -> None:
        if not emit:
            return
        if not think_state["id"]:
            think_state["n"] += 1
            think_state["id"] = f"{stream_id}-t{think_state['n']}"
        await relay_out({"type": "thinking_delta", "stream_id": think_state["id"], "text": chunk, "done": False, "api_session": session_id})

    async def close_thinking(text_: str) -> None:
        if emit and think_state["id"]:
            await relay_out({"type": "thinking_delta", "stream_id": think_state["id"], "done": True, "final_text": text_, "api_session": session_id, "runtime": "api_loop"})
        think_state["id"] = ""

    with_images = vision_enabled()
    messages, had_images = await build_messages(text, atts, before_id=msg_id, session_id=session_id, use_context=use_context, with_images=with_images)
    tools = all_tools()
    max_steps = cfg_int("max_tool_steps", 0, 50)
    texts: list[str] = []
    model_used = ""
    fallback_from: list[str] = []
    error = ""
    step = 0
    try:
        while True:
            try:
                out = await run_model(messages, tools if max_steps > 0 else None, sink=sink, think_sink=think_sink)
            except ModelError as exc:
                # A 400 with pixels in the prompt usually means "this model has no vision": retry text-only once.
                if had_images and exc.status == 400 and with_images:
                    with_images = False
                    messages, had_images = await build_messages(text, atts, before_id=msg_id, session_id=session_id, use_context=use_context, with_images=False)
                    continue
                raise
            calls = out.get("tool_calls") or []
            step_text = out.get("text") or ""
            thinking = out.get("thinking") or ""
            if not step_text and thinking and not calls:
                # Some OpenAI-compatible gateways (seen with Qwen3-VL after a tool result) put the
                # whole final answer in reasoning_content and leave content empty. Promote it.
                step_text, thinking = thinking, ""
            await close_thinking(thinking)
            model_used = out.get("model") or model_used
            fallback_from = out.get("tried") or fallback_from
            if out.get("usage"):
                turn.usage = out["usage"]
            if step_text:
                texts.append(step_text)
            if not calls:
                break
            if step >= max_steps:
                messages.append({"role": "assistant", "content": out.get("text") or None, "tool_calls": [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
                for c in calls:
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": f"ERROR: tool step limit ({max_steps}) reached; answer with what you have."})
                tools = None
                max_steps = 0
                continue
            step += 1
            messages.append({"role": "assistant", "content": out.get("text") or None, "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})
            steps_meta = []
            for c in calls:
                result = await turn.run_tool(c["name"], c["arguments"])
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})
                real = mcp_manager.index.get(c["name"], ("", c["name"]))[1]
                steps_meta.append({"tool": real, "cmd": c["arguments"][:600], "result": result[:1200]})
            act = {"type": "act", "text": _step_label(calls), "glyph": act_glyph(steps_meta[0]["tool"]), "steps": steps_meta, "api_session": session_id, "runtime": "api_loop"}
            turn.acts.append(act)
            if not dry:
                await relay_out(act)
    except ModelError as exc:
        error = exc.detail
    except Exception as exc:  # never leave the PWA stuck on "typing"
        error = f"{type(exc).__name__}: {exc}"

    reply = "\n\n".join(t for t in texts if t).strip()
    if error:
        reply = (reply + "\n\n" if reply else "") + f"⚠️ API loop 出错：{error}"
    if not reply:
        reply = "(The API loop did not produce a reply.)"
    meta: dict[str, Any] = {
        "runtime": "api_loop",
        "model": model_used,
        "fallback_from": fallback_from,
        "usage": turn.usage,
        "session": session_id,
        "tool_steps": step,
    }
    if error:
        meta["error"] = error
    if dry:
        return {"ok": not error, "reply": reply, "api": meta, "acts": turn.acts, "attachments": turn.attachments}
    payload: dict[str, Any] = {"api": meta, "api_session": session_id}
    if turn.attachments:
        payload["attachments"] = turn.attachments
    if emit:
        ok, body = await relay_out({"type": "reply_delta", "stream_id": stream_id, "done": True, "final_text": reply, **payload})
    else:
        ok, body = await relay_out({"type": "reply", "text": reply, **payload})
    return {"ok": ok and not error, "relay": body, "api": meta}


# ---------------------------------------------------------------------------
# config views
# ---------------------------------------------------------------------------

def public_config() -> dict[str, Any]:
    cfg = load_config()
    return {
        "history_n": history_n(),
        "history_images": cfg_int("history_images", 0, 20),
        "max_tool_steps": cfg_int("max_tool_steps", 0, 50),
        "vision": str(cfg.get("vision", CONFIG_DEFAULTS["vision"])),
        "persona": {k: v for k, v in persona_public().items() if k != "text"},
        "active_session": active_session_id(),
        "sessions": session_rows(),
        "main_chain": [
            {"index": i, "model": r.get("model", ""), "url": r.get("url", ""), "key_masked": mask_key(r.get("key", ""))}
            for i, r in enumerate(main_chain())
        ],
        "main_chain_source": "config" if isinstance(cfg.get("main_chain"), list) and cfg.get("main_chain") else ("env" if env_routes() else "none"),
        "mcp_available": MCP_AVAILABLE,
        "mcp_servers": mcp_manager.public(),
        "tools": [t["function"]["name"] for t in all_tools()],
        "stream": STREAM_OUTPUT,
    }


def update_config(body: dict[str, Any]) -> dict[str, Any]:
    cfg = load_config()
    for name, hi in (("history_n", 200), ("history_images", 20), ("max_tool_steps", 50)):
        if name in body:
            try:
                cfg[name] = max(0, min(int(body.get(name) or 0), hi))
            except (TypeError, ValueError):
                raise HTTPException(status_code=400, detail=f"{name} must be an integer")
    if "vision" in body:
        v = str(body.get("vision") or "auto").lower()
        if v not in {"auto", "on", "off"}:
            raise HTTPException(status_code=400, detail="vision must be auto|on|off")
        cfg["vision"] = v
    if "persona_file" in body:
        cfg["persona_file"] = str(body.get("persona_file") or "").strip()
        _persona_cache["mtime"] = None
    if "attach_roots" in body and isinstance(body.get("attach_roots"), list):
        cfg["attach_roots"] = [str(r) for r in body["attach_roots"] if str(r).strip()]
    if isinstance(body.get("main_chain"), list):
        old = main_chain()
        new_chain = []
        for pos, item in enumerate(body["main_chain"]):
            if not isinstance(item, dict):
                continue
            old_idx = int(item.get("index", pos) or 0)
            prev = old[old_idx] if 0 <= old_idx < len(old) else {}
            entry = {
                "model": str(item.get("model") or prev.get("model") or "").strip(),
                "url": str(item.get("url") or prev.get("url") or "").strip().rstrip("/"),
                "key": str(item.get("key") or prev.get("key") or ""),
            }
            if not (entry["model"] and entry["url"] and entry["key"]):
                raise HTTPException(status_code=400, detail=f"row {pos + 1}: model/url/key required")
            new_chain.append(entry)
        cfg["main_chain"] = new_chain  # an empty list falls back to LLM_* env routes
    save_config(cfg)
    return public_config()


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------

@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    LOOP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    await mcp_manager.start_all()
    try:
        yield
    finally:
        await mcp_manager.stop_all()


app = FastAPI(title="companion-api-loop", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "models": [r.get("model") for r in main_chain()],
        "history_n": history_n(),
        "relay_db": RELAY_DB,
        "relay_secret_loaded": bool(RELAY_SECRET),
        "persona_source": persona_public()["source"],
        "mcp": {s["name"]: s["status"] for s in mcp_manager.public()},
        "tools": len(all_tools()),
    }


@app.get("/loop/config")
async def loop_config():
    return public_config()


@app.post("/loop/config")
async def loop_config_update(request: Request):
    return update_config(await request.json())


@app.get("/loop/persona")
async def loop_persona():
    return persona_public()


@app.post("/loop/persona")
async def loop_persona_save(request: Request):
    body = await request.json()
    text = body.get("text")
    if not isinstance(text, str):
        raise HTTPException(status_code=400, detail="text required")
    if len(text) > 200_000:
        raise HTTPException(status_code=413, detail="persona too long")
    return persona_save(text)


@app.get("/loop/tools")
async def loop_tools():
    return {"tools": all_tools(), "count": len(all_tools())}


@app.get("/loop/mcp")
async def loop_mcp_list():
    return {"available": MCP_AVAILABLE, "servers": mcp_manager.public()}


@app.post("/loop/mcp")
async def loop_mcp_upsert(request: Request):
    body = await request.json()
    name = TOOL_NAME_RE.sub("_", str(body.get("name") or "")).strip("_")[:32]
    if not name:
        raise HTTPException(status_code=400, detail="name required (letters, digits, _ -)")
    transport = "http" if str(body.get("transport") or "stdio").lower() in {"http", "streamable-http", "streamable_http"} else "stdio"
    entry: dict[str, Any] = {"name": name, "transport": transport, "enabled": bool(body.get("enabled", True))}
    if transport == "http":
        url = str(body.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            raise HTTPException(status_code=400, detail="http transport needs a url")
        entry["url"] = url
        headers = body.get("headers") or {}
        if isinstance(headers, str):
            try:
                headers = json.loads(headers) if headers.strip() else {}
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="headers must be a JSON object")
        entry["headers"] = {str(k): str(v) for k, v in headers.items()} if isinstance(headers, dict) else {}
    else:
        command = str(body.get("command") or "").strip()
        if not command:
            raise HTTPException(status_code=400, detail="stdio transport needs a command")
        args = body.get("args") or []
        if isinstance(args, str):
            args = [a for a in args.split() if a]
        entry["command"] = command
        entry["args"] = [str(a) for a in args]
        env = body.get("env") or {}
        if isinstance(env, str):
            try:
                env = json.loads(env) if env.strip() else {}
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="env must be a JSON object")
        entry["env"] = {str(k): str(v) for k, v in env.items()} if isinstance(env, dict) else {}
        if body.get("cwd"):
            entry["cwd"] = str(body["cwd"])
    cfg = load_config()
    rows = [r for r in (cfg.get("mcp_servers") or []) if isinstance(r, dict)]
    prev = next((r for r in rows if r.get("name") == name), None)
    if prev:
        # keep masked secrets the UI echoed back unchanged
        for field in ("headers", "env"):
            for k, v in (prev.get(field) or {}).items():
                if entry.get(field, {}).get(k, "") and "***" in entry[field][k]:
                    entry[field][k] = v
        rows[rows.index(prev)] = entry
    else:
        rows.append(entry)
    cfg["mcp_servers"] = rows
    save_config(cfg)
    server = await mcp_manager.restart(name, entry)
    return {"ok": True, "server": server.public() if server else entry, "servers": mcp_manager.public()}


@app.delete("/loop/mcp/{name}")
async def loop_mcp_delete(name: str):
    cfg = load_config()
    rows = [r for r in (cfg.get("mcp_servers") or []) if isinstance(r, dict) and r.get("name") != name]
    cfg["mcp_servers"] = rows
    save_config(cfg)
    old = mcp_manager.servers.pop(name, None)
    if old:
        await old.stop()
    mcp_manager.rebuild_index()
    return {"ok": True, "servers": mcp_manager.public()}


@app.post("/loop/mcp/{name}/reconnect")
async def loop_mcp_reconnect(name: str):
    server = await mcp_manager.restart(name)
    if not server:
        raise HTTPException(status_code=404, detail="server not configured")
    return {"ok": server.status == "online", "server": server.public()}


@app.post("/loop/test")
async def loop_test(request: Request):
    """Round-trip one tiny prompt through a route (index into main_chain, or an ad-hoc url/key/model)."""
    body = await request.json()
    if body.get("url") and body.get("model"):
        idx = int(body.get("index", -1) if str(body.get("index", "")).strip() != "" else -1)
        prev = main_chain()[idx] if 0 <= idx < len(main_chain()) else {}
        key = str(body.get("key") or "")
        if (not key or "***" in key) and prev:
            key = prev.get("key", "")
        route = {"url": str(body["url"]).rstrip("/"), "key": key, "model": str(body["model"])}
    else:
        chain = main_chain()
        idx = int(body.get("index") or 0)
        if not chain or not (0 <= idx < len(chain)):
            raise HTTPException(status_code=400, detail="no such route")
        route = chain[idx]
    started = dt.datetime.now()
    try:
        out = await complete_chat(route, [{"role": "user", "content": "Reply with the single word: pong"}], None)
        ms = int((dt.datetime.now() - started).total_seconds() * 1000)
        return {"ok": True, "model": route["model"], "reply": out.get("text", ""), "latency_ms": ms, "usage": out.get("usage") or {}}
    except ModelError as exc:
        return {"ok": False, "model": route["model"], "error": f"HTTP {exc.status} {exc.detail}"[:600]}
    except Exception as exc:
        return {"ok": False, "model": route["model"], "error": f"{type(exc).__name__}: {exc}"[:600]}


@app.get("/loop/sessions")
async def loop_sessions():
    return sessions_public()


@app.post("/loop/sessions")
async def loop_sessions_create(request: Request):
    body = await request.json()
    row = create_session(
        title=str(body.get("title") or "New chat"),
        since_id=int(body.get("since_id") or 0),
        activate=bool(body.get("activate", True)),
    )
    return {**sessions_public(), "created": row}


@app.patch("/loop/sessions/{session_id}")
async def loop_sessions_patch(session_id: str, request: Request):
    return patch_session(session_id, await request.json())


def _atts_from(body: dict[str, Any], msg_id: int | None) -> list[dict[str, Any]]:
    atts = body.get("attachments")
    if isinstance(atts, list) and atts:
        return [a for a in atts if isinstance(a, dict)]
    return relay_message_attachments(msg_id)


@app.post("/loop/chat")
async def loop_chat(request: Request):
    """Direct chat without the relay round-trip (no streaming, no act chips). Handy for tests."""
    body = await request.json()
    text = str(body.get("text") or body.get("message") or "").strip()
    atts = [a for a in (body.get("attachments") or []) if isinstance(a, dict)]
    if not text and not atts:
        raise HTTPException(status_code=400, detail="empty text")
    session_id = str(body.get("session_id") or body.get("api_session") or active_session_id() or "").strip()
    return await handle_turn(text, atts, None, session_id, dry=True, use_context=bool(body.get("use_context", True)))


@app.post("/loop/ingest")
async def loop_ingest(request: Request):
    body = await request.json()
    text = str(body.get("text") or body.get("message") or "").strip()
    msg_id = body.get("id")
    try:
        before_id = int(msg_id) if msg_id is not None else None
    except Exception:
        before_id = None
    atts = _atts_from(body, before_id)
    if not text and not atts:
        raise HTTPException(status_code=400, detail="empty text")
    session_id = str(body.get("session_id") or body.get("api_session") or active_session_id() or "").strip()
    dry = bool(body.get("dry"))
    return await handle_turn(text, atts, before_id, session_id, dry=dry)


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("LOOP_HOST", "127.0.0.1"), port=LOOP_PORT)
