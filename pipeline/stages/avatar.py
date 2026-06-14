"""Avatar (audio-driven photoreal lip-sync) factory.

The avatar consumes the TTS audio stream and emits a synced video+audio stream.
It is the dominant latency cost, so the prototype uses a managed streaming API.

Providers:
- simli  : low-latency WebRTC avatar, simple API — prototype default.
- heygen : HeyGen LiveAvatar (build against LiveAvatar; Interactive Avatar
           sunsets 2026-03-31).
- musetalk_local : local real-time lip-sync on the 5060 Ti (Phase 3).
"""
from __future__ import annotations

from pathlib import Path

from pipeline.config import Config

# Project root (…/VisualLLm). A one-line file here lets you switch the avatar
# between local MuseTalk and the cloud (Simli) WITHOUT editing .env or restarting
# the pipeline — change it (switch_cloud.bat / switch_local.bat) and reconnect.
_MODE_FILE = Path(__file__).resolve().parents[2] / "avatar_mode.txt"


def _resolve_provider(cfg: Config) -> str:
    """Avatar provider: the avatar_mode.txt override if present, else .env."""
    try:
        val = _MODE_FILE.read_text(encoding="utf-8").strip()
        if val:
            return val
    except Exception:  # noqa: BLE001 — file optional
        pass
    return cfg.avatar_provider


def _warn_if_musetalk_down(base_url: str) -> None:
    """Best-effort pre-flight: warn loudly if the local MuseTalk server isn't up
    yet, so the failure mode is obvious instead of a cryptic websocket error."""
    import json
    import urllib.request

    from loguru import logger

    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/health", timeout=2) as r:
            ok = json.loads(r.read()).get("ok")
        if ok:
            logger.info(f"MuseTalk server is up at {base_url}.")
        else:
            logger.warning(f"MuseTalk server at {base_url} is reachable but not ready.")
    except Exception:  # noqa: BLE001
        logger.warning(
            f"MuseTalk server not reachable at {base_url}. Start it first:\n"
            f"  conda run -n musetalk python -m local_services.musetalk_server.app\n"
            f"(or local_services/musetalk_server/run_server.bat)"
        )


def build_avatar(cfg: Config):
    from loguru import logger

    provider = _resolve_provider(cfg).lower()
    logger.info(f"Avatar provider for this session: {provider}")

    if provider == "simli":
        # Pipecat 1.x takes api_key/face_id directly and builds SimliConfig itself.
        from pipecat.services.simli.video import SimliVideoService

        return SimliVideoService(
            api_key=cfg.simli_api_key,
            face_id=cfg.simli_face_id,
        )

    if provider == "heygen":
        from pipecat.services.heygen.video import HeyGenVideoService

        return HeyGenVideoService(
            api_key=cfg.heygen_api_key,
            avatar_id=cfg.heygen_avatar_id,
        )

    if provider == "musetalk_local":
        from local_services.musetalk_video import MuseTalkVideoService

        _warn_if_musetalk_down(cfg.musetalk_url)
        return MuseTalkVideoService(base_url=cfg.musetalk_url)

    raise ValueError(f"Unknown AVATAR_PROVIDER: {cfg.avatar_provider}")
