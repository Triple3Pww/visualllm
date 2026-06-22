"""Per-process logging setup (shared by both entrypoints).

Each long-running process (the pipeline and the MuseTalk avatar server) calls
`setup_logging("<name>")` once at startup. That gives every run a durable, rotated
file at `logs/<name>.log` capturing all loguru output (Pipecat, our code, the
watchdog) AND the standard-library logging of uvicorn / asyncio / websockets /
onnxruntime -- so when something breaks there is a file to read instead of a guess.

Kept deliberately dependency-light and ASCII-only: it is imported by the MuseTalk
server too, which runs in a separate conda env and must stay ASCII-safe.

Security: the file sink uses diagnose=False on purpose. loguru's diagnose=True
prints local variable values inside tracebacks, which here would include the API
keys held on the config object -- never enable it for these sinks.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from loguru import logger

# logs/ lives next to this module (repo root), so both `python -m` entrypoints
# resolve the same directory regardless of their package.
LOGS_DIR = Path(__file__).resolve().parent / "logs"

# name -> loguru sink id, so we can re-assert a sink that a library (Pipecat's
# runner calls logger.remove() at startup) tore out from under us.
_sink_ids: dict[str, int] = {}


class _InterceptHandler(logging.Handler):
    """Route stdlib logging records into loguru (the standard loguru recipe), so
    uvicorn/asyncio/websockets logs land in the same per-process file."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _add_file_sink(name: str, level: str) -> Path:
    """(Re)add the rotating file sink for `name`, replacing any prior one."""
    LOGS_DIR.mkdir(exist_ok=True)
    logfile = LOGS_DIR / f"{name}.log"
    old = _sink_ids.pop(name, None)
    if old is not None:
        try:
            logger.remove(old)
        except ValueError:
            pass  # already gone (e.g. Pipecat's logger.remove() took it)
    _sink_ids[name] = logger.add(
        str(logfile),
        level=level,
        rotation="10 MB",      # roll the file once it grows past 10 MB
        retention=10,          # keep the last 10 rolled files
        enqueue=True,          # process/thread-safe (own writer thread)
        backtrace=True,        # full tracebacks for crashes
        diagnose=False,        # SECURITY: never dump variable values (API keys)
        encoding="utf-8",
    )
    # force=True replaces any handlers a library installed so records reach us.
    # level=INFO (not 0/NOTSET): aiortc/aioice emit a DEBUG record PER RTP PACKET, and
    # every stdlib record runs _InterceptHandler.emit (a stack-walk) on the realtime
    # asyncio media loop before loguru can filter it -- at 0 that floods 10MB/20min AND
    # starves the loop that reads the inbound mic track (-> "Media stream error; clearing
    # track"). Gating at the stdlib root drops those packet records at the source. Our own
    # loguru calls bypass stdlib, so this does NOT touch our app's logging detail.
    logging.basicConfig(handlers=[_InterceptHandler()], level=logging.INFO, force=True)
    # Belt-and-suspenders: pin the known per-packet spammers to WARNING regardless of root.
    for _noisy in ("aiortc", "aioice", "aiortc.rtcrtpsender", "aiortc.rtcrtpreceiver"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    return logfile


def setup_logging(name: str, level: str = "DEBUG") -> Path:
    """Add a rotating file sink at logs/<name>.log and intercept stdlib logging.

    Returns the log file path. The existing console sink is left in place so
    interactive runs still print to the terminal.
    """
    logfile = _add_file_sink(name, level)
    logger.info(f"=== {name} started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    return logfile


def ensure_file_sink(name: str, level: str = "DEBUG") -> Path:
    """Re-assert the file sink after a library tore it out.

    Pipecat's runner calls ``logger.remove()`` when ``main()`` starts, which drops
    the sink added in the process entrypoint -- so logs/<name>.log would otherwise
    miss every runtime log. Call this once the runner is up (e.g. in run_bot) to
    put the file sink back.
    """
    return _add_file_sink(name, level)
