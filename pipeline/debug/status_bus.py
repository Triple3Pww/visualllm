"""StatusBus — the single in-process source of truth for the debug dashboard.

Pure data + health logic; deliberately NO Pipecat imports (the version-sensitive
frame knowledge lives in taps.py, which translates frames into the semantic calls
below). The `StageTap`s write here; the debug web server reads `snapshot()`.

Everything runs in the pipeline's single asyncio loop thread, so no locking is
needed: taps, the /status poller, and the websocket sender never run concurrently.

The whole point is the colour logic in `snapshot()` — it turns raw activity into a
per-stage green / yellow / red so "something is broken but I don't know where"
becomes "the AVATAR light is red: server unreachable".
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

# Stage chain, in pipeline order. (key, label, sublabel-default)
STAGE_DEFS: list[tuple[str, str, str]] = [
    ("vad", "VAD", "Silero"),
    ("stt", "STT", "Deepgram"),
    ("llm", "LLM", "OpenRouter"),
    ("tts", "TTS", "ElevenLabs"),
    ("avatar", "AVATAR", "Ditto"),
    ("out", "out", "WebRTC"),
]
STAGE_KEYS = [s[0] for s in STAGE_DEFS]

# A stage glows green if it produced output within this many seconds.
ACTIVE_WINDOW_S = 1.0
# During an active turn, if a stage's upstream neighbour has produced but this
# stage still hasn't after this long, call it red ("expected output, got none").
STAGE_DEADLINE_S = {"stt": 6.0, "llm": 9.0, "tts": 9.0, "avatar": 12.0}
# The "out" stage is judged by the bot actually starting to speak (BotStarted).


def _now() -> float:
    return time.monotonic()


def _clock() -> str:
    return time.strftime("%H:%M:%S")


class StatusBus:
    def __init__(self) -> None:
        # The dashboard server runs in its own daemon thread (so it's up from
        # process start, before any client connects), while the taps write from
        # the pipeline's event-loop thread. This RLock guards the collections that
        # both threads touch (the ring buffers / TTFO list) so a snapshot can't
        # iterate a deque mid-append. Reentrant: on_bot_started/update_avatar log.
        self._lock = threading.RLock()
        # Per-stage live state.
        self._stage_label = {k: lbl for k, lbl, _ in STAGE_DEFS}
        self._stage_sub = {k: sub for k, _, sub in STAGE_DEFS}
        self._last_output: dict[str, float] = {k: 0.0 for k in STAGE_KEYS}
        self._error: dict[str, str | None] = {k: None for k in STAGE_KEYS}
        self._error_t: dict[str, float] = {k: 0.0 for k in STAGE_KEYS}

        # Turn tracking.
        self._turn_id = 0
        self._turn_start: float | None = None  # set on UserStopped; None between turns
        self._first_output: dict[str, float | None] = {k: None for k in STAGE_KEYS}

        # TTFO samples + recent completed turns (for the timeline rows).
        self._ttfo: list[float] = []
        self._turns: deque[dict] = deque(maxlen=8)
        self._events: deque[dict] = deque(maxlen=60)

        # Avatar: server /status snapshot (poller) + a ref to the live client.
        self._avatar_raw: dict | None = None
        self._avatar_reachable = False
        self._avatar_seen_t = 0.0
        self._avatar_service: Any = None  # DittoVideoService instance, read defensively

        # Static config the colour logic needs.
        self._avatar_enabled = True
        self._fps_target = 25
        self._ttfo_target_s = 8.0
        self._lang = "en"
        self._started_t = _now()

    # --- configuration (called once from main.py) -------------------------
    def configure(self, *, avatar_enabled: bool, fps_target: int,
                  ttfo_target_s: float, lang: str) -> None:
        self._avatar_enabled = avatar_enabled
        self._fps_target = fps_target
        self._ttfo_target_s = ttfo_target_s
        self._lang = lang

    def set_avatar_service(self, svc: Any) -> None:
        self._avatar_service = svc

    # --- semantic events (called by StageTap) -----------------------------
    def on_user_stopped(self) -> None:
        """User finished a turn — start the clock. Idempotent within a turn."""
        if self._turn_start is not None:
            return  # already mid-turn; ignore the duplicate frame
        self._turn_id += 1
        self._turn_start = _now()
        self._first_output = {k: None for k in STAGE_KEYS}

    def on_bot_started(self) -> None:
        """Bot's first output — the TTFO stop event. Closes the current turn."""
        with self._lock:
            if self._turn_start is None:
                return  # e.g. the connect greeting (no preceding user turn): ignore
            ttfo = _now() - self._turn_start
            self._ttfo.append(ttfo)
            over = ttfo > self._ttfo_target_s
            self._turns.appendleft({
                "id": self._turn_id,
                "ttfo": round(ttfo, 2),
                "over": over,
                "breakdown": self._turn_breakdown(),
            })
            if over:
                self.log("warn", f"turn #{self._turn_id} TTFO {ttfo:0.1f}s OVER "
                                 f"(>{self._ttfo_target_s:0.0f}s) - {self._slowest_stage()}")
            self._turn_start = None  # turn closed; stages idle until next UserStopped

    def mark_output(self, stage: str) -> None:
        """A stage produced its characteristic output frame."""
        t = _now()
        self._last_output[stage] = t
        # Clear a stale error once the stage is visibly working again.
        if self._error[stage] is not None and t - self._error_t[stage] > 1.0:
            self._error[stage] = None
        if self._turn_start is not None and self._first_output[stage] is None:
            self._first_output[stage] = t

    def mark_error(self, stage: str, msg: str) -> None:
        self._error[stage] = msg
        self._error_t[stage] = _now()
        self.log("error", f"{self._stage_label.get(stage, stage)}: {msg}")

    def log(self, level: str, msg: str) -> None:
        with self._lock:
            self._events.appendleft({"t": _clock(), "level": level, "msg": msg})

    # --- avatar server status (called by the poller) ----------------------
    def update_avatar(self, raw: dict | None, reachable: bool) -> None:
        with self._lock:
            was = self._avatar_reachable
            self._avatar_raw = raw
            self._avatar_reachable = reachable
            self._avatar_seen_t = _now()
            if self._avatar_enabled and was and not reachable:
                self.log("error", "AVATAR: Ditto server became unreachable")
            elif self._avatar_enabled and not was and reachable:
                self.log("info", "AVATAR: Ditto server reachable")

    # --- internals --------------------------------------------------------
    def _turn_breakdown(self) -> dict[str, float | None]:
        """Per-stage incremental latency (seconds) for the just-finished turn."""
        out: dict[str, float | None] = {}
        prev = self._turn_start
        for k in ("stt", "llm", "tts", "avatar"):
            fo = self._first_output[k]
            if fo is not None and prev is not None:
                out[k] = round(fo - prev, 2)
                prev = fo
            else:
                out[k] = None
        return out

    def _slowest_stage(self) -> str:
        bd = self._turn_breakdown()
        best, val = None, -1.0
        for k, v in bd.items():
            if v is not None and v > val:
                best, val = k, v
        return f"slowest: {self._stage_label.get(best, best)} {val:0.1f}s" if best else "no breakdown"

    def _stage_latency_ms(self, stage: str) -> int | None:
        """Live incremental latency this turn, if the stage has produced yet."""
        if self._turn_start is None:
            # Fall back to the most recent completed turn so numbers don't vanish.
            if self._turns:
                v = self._turns[0]["breakdown"].get(stage)
                return int(v * 1000) if v is not None else None
            return None
        order = ["stt", "llm", "tts", "avatar"]
        if stage not in order:
            return None
        fo = self._first_output[stage]
        if fo is None:
            return None
        idx = order.index(stage)
        prev = self._turn_start if idx == 0 else self._first_output[order[idx - 1]]
        if prev is None:
            return None
        return int((fo - prev) * 1000)

    def _stage_state(self, stage: str) -> tuple[str, str | None]:
        """(state, reason) where state is green | yellow | red."""
        t = _now()
        if self._error[stage] is not None and t - self._error_t[stage] < 5.0:
            return "red", self._error[stage]
        # Expected-but-missing: upstream produced this turn, this stage didn't.
        if self._turn_start is not None and stage in STAGE_DEADLINE_S:
            order = ["stt", "llm", "tts", "avatar"]
            idx = order.index(stage)
            upstream_ok = idx == 0 or self._first_output[order[idx - 1]] is not None
            waited = t - self._turn_start
            if (upstream_ok and self._first_output[stage] is None
                    and waited > STAGE_DEADLINE_S[stage]):
                return "red", f"no output after {waited:0.1f}s"
        if t - self._last_output[stage] <= ACTIVE_WINDOW_S:
            return "green", None
        return "yellow", None

    def _avatar_panel(self) -> dict:
        """The deep panel + the AVATAR strip light. Maps the documented Ditto
        failure modes (unreachable / CPU fallback / stalled / desynced) to red."""
        svc = self._avatar_service
        raw = self._avatar_raw or {}
        unsynced = bool(getattr(svc, "_unsynced", False)) if svc else False
        lead_s = float(getattr(svc, "_lead_s", 0.2)) if svc else 0.2
        ws_connected = (getattr(svc, "_ws", None) is not None) if svc else False
        fps = float(raw.get("fps", 0.0) or 0.0)
        real_fps = raw.get("real_fps")
        target = int(raw.get("fps_target", self._fps_target) or self._fps_target)
        queue = int(raw.get("queue", 0) or 0)
        speaking = bool(raw.get("speaking", False))
        workers_stopped = bool(raw.get("workers_stopped", False))
        onnx_cuda = raw.get("onnx_cuda")
        torch_cuda = raw.get("torch_cuda")
        session_active = bool(raw.get("session_active", False))
        gpu_ok = bool(onnx_cuda) if onnx_cuda is not None else None

        # Fresh enough? The poller stamps _avatar_seen_t; stale => treat unreachable.
        reachable = self._avatar_reachable and (_now() - self._avatar_seen_t < 4.0)

        if not self._avatar_enabled:
            state, reason = "off", "audio-only (client renders the face)"
        elif not reachable:
            state, reason = "red", "server unreachable"
        elif workers_stopped:
            state, reason = "red", "render workers crashed"
        elif session_active and gpu_ok is False:
            state, reason = "red", "GPU not active - CPU fallback (~5x slow)"
        elif unsynced:
            state, reason = "red", "sync FALLBACK (no video clock)"
        elif speaking and fps > 0 and fps < 0.6 * target:
            state, reason = "red", f"render stalled ({fps:0.0f}/{target} fps)"
        elif speaking:
            state, reason = "green", "rendering"
        elif reachable:
            state, reason = "yellow", "idle (reachable)"
        else:
            state, reason = "yellow", None

        return {
            "state": state,
            "reason": reason,
            "reachable": reachable,
            "ws_connected": ws_connected,
            "gpu_ok": gpu_ok,
            "torch_cuda": torch_cuda,
            "fps": round(fps, 1),
            "real_fps": real_fps,  # true rendered-frame rate (vs the padded `fps`)
            "fps_target": target,
            "queue": queue,
            "speaking": speaking,
            "workers_stopped": workers_stopped,
            "session_active": session_active,
            "sync": "FALLBACK" if unsynced else "SYNCED",
            "lead_s": round(lead_s, 2),
            # Render headroom (from the Ditto watchdog) for tuning delay/lag.
            # gpu_util is the real diffusion-load signal; feat_* is hubert extraction.
            "gpu_util": raw.get("gpu_util"),
            "vram_used": raw.get("vram_used"),
            "vram_total": raw.get("vram_total"),
            "gpu_temp": raw.get("gpu_temp"),
            "feat_avg_ms": raw.get("feat_avg_ms"),
            "feat_max_ms": raw.get("feat_max_ms"),
            "feat_budget_ms": raw.get("feat_budget_ms"),
            "q_peak": raw.get("q_peak"),
        }

    def _ttfo_block(self) -> dict:
        if not self._ttfo:
            return {"count": 0, "target_s": self._ttfo_target_s}
        s = sorted(self._ttfo)
        p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
        return {
            "count": len(s),
            "last_s": round(self._ttfo[-1], 2),
            "p95_s": round(p95, 2),
            "median_s": round(s[len(s) // 2], 2),
            "target_s": self._ttfo_target_s,
            "pass": p95 <= self._ttfo_target_s,
        }

    # --- the snapshot the dashboard renders -------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict:
        avatar = self._avatar_panel()
        stages = []
        for key in STAGE_KEYS:
            if key == "avatar":
                state, reason = avatar["state"], avatar["reason"]
            else:
                state, reason = self._stage_state(key)
            stages.append({
                "key": key,
                "label": self._stage_label[key],
                "sub": self._stage_sub[key],
                "state": state,
                "reason": reason,
                "latency_ms": self._stage_latency_ms(key),
            })
        return {
            "t": _clock(),
            "uptime_s": int(_now() - self._started_t),
            "lang": self._lang,
            "turn": self._turn_id,
            "turn_active": self._turn_start is not None,
            "stages": stages,
            "avatar": avatar,
            "ttfo": self._ttfo_block(),
            "turns": list(self._turns),
            "events": list(self._events),
        }


# Module-level singleton shared by taps and the debug server.
bus = StatusBus()
