"""Avatar: local audio-driven lip-sync running on the GPU (MuseTalk).

The avatar consumes the TTS audio stream and emits a synced video+audio stream.
It is the dominant latency cost, so it runs locally (no cloud round-trip).

MuseTalk runs as a separate GPU server (local_services/musetalk_server/, the
`musetalk` conda env) on port 8002: mouth-region lip-sync, no warmup, sharp lips,
driven by a female portrait via AVATAR_REF. This client talks to it over cfg.avatar_url.
"""
from __future__ import annotations

from pipeline.config import Config


def build_avatar(cfg: Config):
    from loguru import logger

    from local_services.musetalk_video import MuseTalkVideoService

    # OUTPUT fps the server pushes (config.avatar_fps, MuseTalk ~20). The server
    # frame-drops to this so a sub-realtime GPU stays realtime. INTEGER fps so the
    # server push rate == the transport's video_out_framerate EXACTLY (main.py couples
    # both to config.avatar_fps; a mismatch slowly desyncs A/V).
    fps = cfg.avatar_fps
    # Frame size must match the server's size env and main.py's video_out_*; the service
    # tags every frame with this, so a wrong value hands aiortc bad dims.
    size = cfg.avatar_size
    _warn_if_server_down(cfg.avatar_url)

    logger.info(f"Avatar: local MuseTalk at {cfg.avatar_url} (output fps={fps}, size={size})")
    return MuseTalkVideoService(base_url=cfg.avatar_url, fps=int(fps), image_size=(size, size))


def _warn_if_server_down(base_url: str) -> None:
    """Best-effort pre-flight: warn loudly if the local avatar server isn't up
    yet, so the failure mode is obvious instead of a cryptic websocket error."""
    import json
    import urllib.request

    from loguru import logger

    start_hint = "  conda run -n musetalk python -m local_services.musetalk_server.app"

    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=2) as r:
            ok = json.loads(r.read()).get("ok")
        if ok:
            logger.info(f"Avatar server is up at {base_url}.")
        else:
            logger.warning(f"Avatar server at {base_url} is reachable but not ready.")
    except Exception:  # noqa: BLE001
        logger.warning(
            f"Avatar server not reachable at {base_url}. Start it first:\n{start_hint}"
        )
