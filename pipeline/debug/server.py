"""Debug dashboard web server.

A small FastAPI app that runs as an asyncio task *inside the pipeline process*
(so it shares the in-process StatusBus) on its own port — fully decoupled from
Pipecat's own WebRTC server, so it can never affect the pipeline. It:

  GET /debug (or /)  -> the dashboard HTML (Layout A)
  WS  /ws            -> streams `bus.snapshot()` ~5 Hz
  background task    -> polls the Ditto server's GET /status into the bus

Start it once from main.py via `start_debug_server(...)`.
"""
from __future__ import annotations

import asyncio
import json
import threading
import urllib.request
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, PlainTextResponse
from loguru import logger

from pipeline.debug.status_bus import bus

_HTML_PATH = Path(__file__).parent / "dashboard.html"
# logs/ lives at the repo root (two levels up from pipeline/debug/).
_LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"
# Whitelist so /logs/{name} can't be used to read arbitrary files.
_LOG_FILES = {"pipeline": "pipeline.log", "ditto": "ditto.log"}

app = FastAPI(title="VisualLLm debug")


@app.get("/")
@app.get("/debug")
async def dashboard() -> HTMLResponse:
    return HTMLResponse(_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/logs/{name}")
async def logs(name: str, lines: int = 300) -> PlainTextResponse:
    """Tail the last `lines` of a process log so the dashboard can show why a
    stage went red. Both files are local even though Ditto is another process."""
    fname = _LOG_FILES.get(name)
    if fname is None:
        return PlainTextResponse(f"(unknown log '{name}')", status_code=404)
    path = _LOGS_DIR / fname
    if not path.exists():
        return PlainTextResponse(f"(no log yet at {path})")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return PlainTextResponse(f"(could not read {path}: {e})", status_code=500)
    tail = text.splitlines()[-max(1, min(lines, 5000)):]
    return PlainTextResponse("\n".join(tail))


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(bus.snapshot())
            await asyncio.sleep(0.2)
    except Exception:  # noqa: BLE001 — client closed / navigated away
        pass


def _http_get_json(url: str, timeout: float = 1.5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


async def _poll_avatar(avatar_url: str) -> None:
    """Pull the Ditto server's live metrics into the bus. urllib in a thread keeps
    it dependency-free (same approach as the avatar stage's /health pre-check)."""
    url = avatar_url.rstrip("/") + "/status"
    while True:
        try:
            raw = await asyncio.to_thread(_http_get_json, url)
            bus.update_avatar(raw, reachable=True)
        except Exception:  # noqa: BLE001 — server down/starting is a normal state
            bus.update_avatar(None, reachable=False)
        await asyncio.sleep(0.5)


async def _serve(port: int, avatar_url: str, avatar_enabled: bool) -> None:
    if avatar_enabled:
        asyncio.create_task(_poll_avatar(avatar_url))
    cfg = uvicorn.Config(
        app, host="0.0.0.0", port=port, log_level="warning",
        ws_ping_interval=None, ws_ping_timeout=None,
        log_config=None,  # let records propagate into loguru (logs/pipeline.log)
    )
    server = uvicorn.Server(cfg)
    # Signal handlers only install in the main thread; we run in a daemon thread.
    server.install_signal_handlers = lambda: None  # type: ignore[assignment]
    await server.serve()


_started = False


def start_debug_server(*, port: int, avatar_url: str, avatar_enabled: bool) -> None:
    """Idempotently launch the dashboard server in its OWN daemon thread + loop.

    A separate thread (not the pipeline's event loop) means the dashboard is
    reachable from process start — before any /client session connects — and is
    fully decoupled from the pipeline. The StatusBus it reads is guarded by a lock
    for this cross-thread access.
    """
    global _started
    if _started:
        return
    _started = True

    def _run() -> None:
        asyncio.run(_serve(port, avatar_url, avatar_enabled))

    threading.Thread(target=_run, name="debug-dashboard", daemon=True).start()
    logger.info(f"Debug dashboard: http://localhost:{port}/debug")
