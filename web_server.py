"""
CoreCoder Web Server — FastAPI + SSE streaming + multi-user session management.

Usage:
    # 配好 vLLM 地址
    set OPENAI_BASE_URL=http://your-vllm-server:8000/v1
    set OPENAI_API_KEY=vllm
    set CORECODER_MODEL=Qwen/Qwen3-32B

    # 启动
    python web_server.py --port 8080

⚠  vLLM 必须用以下参数启动才支持工具调用:
    vllm serve <model> --enable-auto-tool-choice --tool-call-parser hermes
"""

import json
import asyncio
import uuid
import time
import argparse
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corecoder.agent import Agent
from corecoder.llm import LLM
from corecoder.config import Config
from corecoder.tools.docx import OUTPUT_DIR as DOCX_OUTPUT_DIR


# ────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────
def _brief_args(args: dict, maxlen: int = 80) -> str:
    """Compact representation of tool arguments for the UI."""
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 40:
            s = s[:37] + "..."
        parts.append(f"{k}={s}")
    joined = ", ".join(parts)
    return joined[:maxlen]


def _safe_put(queue: asyncio.Queue, item):
    """Put item on asyncio.Queue, dropping if full. Must run on the event loop thread."""
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        pass


# ────────────────────────────────────────────────────────────
#  WebAgent — emits tool events via a thread-safe callable
# ────────────────────────────────────────────────────────────
class WebAgent(Agent):
    """Agent that pushes tool-start/tool-result events for SSE streaming.

    The emit callable wraps loop.call_soon_threadsafe, making it safe to call
    from the ThreadPoolExecutor thread where agent.chat() runs.

    Both tool_start and tool_result carry the tool_call id so the frontend
    can match results to the right tool card even when tools run in parallel.
    """

    def __init__(self, *args, emit=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._emit = emit
        self._cancelled = False

    def cancel(self):
        """Signal the agent to stop executing further tools (client disconnect)."""
        self._cancelled = True

    def _exec_tool(self, tc):
        if self._cancelled:
            return "[cancelled by client disconnect]"

        if self._emit:
            self._emit({
                "type": "tool_start",
                "id": tc.id,
                "name": tc.name,
                "args": _brief_args(tc.arguments),
            })

        result = super()._exec_tool(tc)

        if self._emit:
            preview = result[:600] + ("…" if len(result) > 600 else "")
            self._emit({
                "type": "tool_result",
                "id": tc.id,
                "name": tc.name,
                "preview": preview,
            })
        return result


# ────────────────────────────────────────────────────────────
#  Session Manager
# ────────────────────────────────────────────────────────────
class SessionManager:
    def __init__(self):
        self._sessions: dict[str, dict] = {}

    def create(self, llm: LLM, max_context_tokens: int = 128_000) -> str:
        sid = uuid.uuid4().hex[:12]
        self._sessions[sid] = {
            "agent": None,
            "llm": llm,
            "max_context_tokens": max_context_tokens,
            "model": llm.model,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "title": "New session",
        }
        return sid

    def get_or_create_agent(self, sid: str, emit) -> WebAgent:
        """Return (or lazily create) the WebAgent for a session, wired to emit."""
        entry = self._sessions.get(sid)
        if entry is None:
            raise KeyError(sid)

        if entry["agent"] is None:
            entry["agent"] = WebAgent(
                llm=entry["llm"],
                max_context_tokens=entry["max_context_tokens"],
            )
        # swap the emit callable to the current request's queue/loop
        entry["agent"]._emit = emit
        entry["agent"]._cancelled = False

        if entry["title"] == "New session" and entry["agent"].messages:
            first = next(
                (m["content"][:40] for m in entry["agent"].messages if m["role"] == "user"),
                None,
            )
            if first:
                entry["title"] = first

        return entry["agent"]

    def get(self, sid: str) -> dict | None:
        return self._sessions.get(sid)

    def delete(self, sid: str):
        self._sessions.pop(sid, None)

    def list_sessions(self) -> list[dict]:
        return [
            {
                "id": sid,
                "model": s["model"],
                "created_at": s["created_at"],
                "title": s["title"],
                "message_count": len(s["agent"].messages) if s["agent"] else 0,
            }
            for sid, s in self._sessions.items()
        ]

    def cleanup_stale(self, max_age_seconds: int = 3600):
        """Remove sessions older than max_age_seconds."""
        now = time.time()
        stale = []
        for sid, s in self._sessions.items():
            try:
                created = time.mktime(time.strptime(s["created_at"], "%Y-%m-%d %H:%M:%S"))
                if now - created > max_age_seconds:
                    stale.append(sid)
            except Exception:
                pass
        for sid in stale:
            self._sessions.pop(sid, None)


# ── global state ────────────────────────────────────────────
sessions = SessionManager()
config = Config.from_env()
_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    return _llm


# ────────────────────────────────────────────────────────────
#  Lifespan: periodic session cleanup
# ────────────────────────────────────────────────────────────
async def _periodic_cleanup():
    while True:
        await asyncio.sleep(300)
        try:
            sessions.cleanup_stale(max_age_seconds=3600)
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(_periodic_cleanup())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


# ── app setup ───────────────────────────────────────────────
app = FastAPI(title="CoreCoder Web", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=16)


# ────────────────────────────────────────────────────────────
#  Pydantic models
# ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str


class SessionCreate(BaseModel):
    model: str | None = None


# ────────────────────────────────────────────────────────────
#  API Routes
# ────────────────────────────────────────────────────────────

@app.post("/api/sessions")
def create_session(body: SessionCreate | None = None):
    """Create a new chat session, return its ID."""
    llm = get_llm()
    if body and body.model:
        llm = LLM(
            model=body.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
    sid = sessions.create(llm, max_context_tokens=config.max_context_tokens)
    return {"session_id": sid, "model": llm.model}


@app.get("/api/sessions")
def list_sessions():
    """List all active sessions."""
    return sessions.list_sessions()


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    """End a session and free its resources."""
    sessions.delete(session_id)
    return {"ok": True}


@app.get("/api/sessions/{session_id}/history")
def get_history(session_id: str):
    """Return raw message history for a session."""
    entry = sessions.get(session_id)
    if entry is None:
        raise HTTPException(404, "Session not found")
    agent = entry["agent"]
    if agent is None:
        return {"messages": [], "model": entry["model"]}
    return {"messages": agent.messages, "model": entry["model"]}


@app.get("/api/health")
def health():
    return {"ok": True, "model": config.model}


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Serve generated Word documents for download."""
    # sanitize: only alnum, dash, underscore, dot — blocks path traversal
    safe = "".join(c for c in filename if c.isalnum() or c in "-_.")
    if not safe.endswith(".docx"):
        raise HTTPException(400, "Only .docx files are available for download")
    path = (DOCX_OUTPUT_DIR / safe).resolve()
    if not path.is_relative_to(DOCX_OUTPUT_DIR.resolve()):
        raise HTTPException(400, "Invalid file path")
    if not path.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(
        path,
        filename=safe,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@app.post("/api/chat/{session_id}")
async def chat(session_id: str, body: ChatRequest):
    """Send a message and stream the response via SSE.

    SSE event types:
      token       — a piece of streaming text
      tool_start  — agent is about to run a tool (carries id)
      tool_result — tool execution finished (carries id to match tool_start)
      done        — turn complete
      error       — something went wrong
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)
    loop = asyncio.get_running_loop()

    # thread-safe emit: schedules the put on the event loop thread
    # (asyncio.Queue is NOT thread-safe — direct put_nowait from the
    # executor thread races with the event loop's getter futures)
    def emit(event):
        loop.call_soon_threadsafe(_safe_put, queue, event)

    try:
        agent = sessions.get_or_create_agent(session_id, emit)
    except KeyError:
        raise HTTPException(404, "Session not found")

    def on_token(tok: str):
        emit({"type": "token", "content": tok})

    # tool events are emitted by WebAgent._exec_tool itself (with tc.id),
    # so we don't pass on_tool here — avoids duplicate/unpaired events.

    async def run():
        try:
            result = await loop.run_in_executor(
                _executor,
                lambda: agent.chat(body.message, on_token=on_token),
            )
            queue.put_nowait({"type": "done", "content": result})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            queue.put_nowait({"type": "error", "message": str(exc)})
        finally:
            queue.put_nowait(None)  # sentinel → generator stops

    task = asyncio.create_task(run())

    async def event_stream():
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            # client disconnected (or stream ended) — stop the agent to
            # avoid burning tokens on a response nobody will see
            agent.cancel()
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ────────────────────────────────────────────────────────────
#  Static files
# ────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


@app.get("/")
async def index():
    """Serve the chat UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>static/index.html not found.</h2>")


# ────────────────────────────────────────────────────────────
#  Entry point
# ────────────────────────────────────────────────────────────
def main():
    global config
    parser = argparse.ArgumentParser(description="CoreCoder Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--base-url", help="Override API base URL")
    parser.add_argument("--api-key", help="Override API key")
    args = parser.parse_args()

    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key

    if not config.api_key:
        print("⚠  No API key — vLLM doesn't need one, but set OPENAI_API_KEY=anything")
        config.api_key = "vllm"

    print(f"""
══════════════════════════════════════════════
  CoreCoder Web Server
  Model : {config.model}
  Base  : {config.base_url}
  Addr  : http://{args.host}:{args.port}
══════════════════════════════════════════════
  ⚠ vLLM 工具调用必须用以下参数启动:
    vllm serve <model> --enable-auto-tool-choice --tool-call-parser hermes
  否则 agent 的工具永远不会触发
══════════════════════════════════════════════
""".strip())

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
