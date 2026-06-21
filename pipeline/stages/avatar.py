"""Avatar: local audio-driven lip-sync running on the GPU (MuseTalk or Ditto).

The avatar consumes the TTS audio stream and emits a synced video+audio stream.
It is the dominant latency cost, so it runs locally (no cloud round-trip).

Both engines run as a separate GPU server sharing ONE wire contract and port (8002):
  * MuseTalk (default) -- local_services/musetalk_server/, `musetalk` conda env;
    mouth-region lip-sync, no warmup, sharper lips, but only the mouth animates.
  * Ditto (fallback)   -- local_services/ditto_server/, `ditto` conda env;
    full-face talking-head, TensorRT-accelerated, ~2.2s diffusion warmup.
This client talks to whichever is running over `cfg.avatar_url`.
"""
from __future__ import annotations

from pipeline.config import Config


def build_avatar(cfg: Config):
    from loguru import logger

    # OUTPUT fps the server pushes (engine-aware via config.avatar_fps: MuseTalk ~20,
    # Ditto ~12). The server frame-drops to this so a sub-realtime GPU stays realtime.
    # INTEGER fps so the server push rate == the transport's video_out_framerate EXACTLY
    # (main.py couples both to config.avatar_fps; a mismatch slowly desyncs A/V).
    fps = cfg.avatar_fps
    # Frame size must match the server's size env and main.py's video_out_*; the service
    # tags every frame with this, so a wrong value hands aiortc bad dims.
    size = cfg.avatar_size
    _warn_if_server_down(cfg.avatar_url, cfg.avatar_mode)

    if cfg.avatar_mode == "musetalk":
        from local_services.musetalk_video import MuseTalkVideoService

        logger.info(f"Avatar: local MuseTalk at {cfg.avatar_url} (output fps={fps}, size={size})")
        return MuseTalkVideoService(base_url=cfg.avatar_url, fps=int(fps), image_size=(size, size))

    from local_services.ditto_video import DittoVideoService

    logger.info(f"Avatar: local Ditto at {cfg.avatar_url} (output fps={fps}, size={size})")
    return DittoVideoService(base_url=cfg.avatar_url, fps=float(fps), image_size=(size, size))


def _warn_if_server_down(base_url: str, avatar_mode: str = "musetalk") -> None:
    """Best-effort pre-flight: warn loudly if the local avatar server isn't up
    yet, so the failure mode is obvious instead of a cryptic websocket error."""
    import json
    import urllib.request

    from loguru import logger

    if avatar_mode == "musetalk":
        start_hint = "  conda run -n musetalk python -m local_services.musetalk_server.app"
    else:
        start_hint = "  conda run -n ditto python -m local_services.ditto_server.app"

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
