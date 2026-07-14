"""Pipeline assembly + runner.

Order of frames per turn:
  mic -> transport.input() -> [Silero VAD on input] -> STT -> user context
       -> LLM (streamed, sentence-aggregated) -> TTS -> Avatar(lip-sync)
       -> TtfoMeter -> transport.output() -> browser (video+audio)

Run locally:
  python -m pipeline.main
then open the printed http://localhost URL in a browser.

This targets a recent Pipecat (uses the development runner + SmallWebRTC). If an
import path errors, check it against your installed version — the fragile bits
are isolated to the stage factories and the imports at the top here.
"""
from __future__ import annotations

import asyncio
import os
import time

from loguru import logger

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.runner.types import RunnerArguments
from pipecat.transports.base_transport import BaseTransport, TransportParams

from pipeline.config import config
from pipeline.metrics import TtfoMeter
from pipeline.stages import build_avatar, build_llm, build_stt, build_tts, build_vad_params
from local_services.avatar_memory import MemoryStore


async def run_bot(transport: BaseTransport, conn=None) -> None:
    # Pipecat's runner removes loguru sinks when main() starts, dropping the file
    # sink added in __main__ -> logs/pipeline.log would miss all runtime logs. This
    # runs after the runner has configured logging, so it re-asserts the file sink.
    from log_setup import ensure_file_sink

    ensure_file_sink("pipeline")

    _llm_label = "WeatherChain" if config.llm_provider == "weather_chain" else "OpenRouter"
    logger.info(
        f"Pipeline: Deepgram STT -> {_llm_label} LLM -> CosyVoice TTS -> MuseTalk avatar "
        f"(lang={config.language})"
    )

    stt = build_stt(config)
    # Memory harness: only for the weather bot, only when enabled. Wrapped around the
    # stateless chain (the chain can't hold memory); rewrites the query + distills the
    # conversation. Local qwen on CPU, so the GPU stays free for the avatar.
    memory = None
    if config.llm_provider == "weather_chain" and config.avatar_memory:
        memory = MemoryStore(
            base_dir=config.avatar_memory_dir,
            llm_url=config.memory_llm_url,
            llm_model=config.memory_llm_model,
            gated=config.memory_llm_gated,
            enabled=True,
        )
        logger.info(
            f"Avatar memory ON (model={config.memory_llm_model}, gated={config.memory_llm_gated})."
        )
        # Startup recovery: fold in any turns a crashed prior session left behind
        # (instant no-op on a normal boot). Runs before any client connects, so the
        # next conversation starts from a clean session.
        await memory.distill_pending()
    llm = build_llm(config, memory)
    tts = build_tts(config)
    avatar = build_avatar(config)
    meter = TtfoMeter(target_s=config.ttfo_target_s)

    context = LLMContext([{"role": "system", "content": config.system_prompt}])
    # Two independent, optional tweaks to the user aggregator, both via LLMUserAggregatorParams:
    #   * Echo-guard (ECHO_GUARD=1): mute the mic while the bot speaks (half-duplex) via
    #     AlwaysUserMuteStrategy. BROKEN under steady sync (P11) -> default OFF.
    #   * No-interrupt (ALLOW_INTERRUPTIONS=0): the bot always finishes its turn; user speech
    #     during playback never cancels it. Done by turning OFF `enable_interruptions` on the
    #     default turn-START strategies (the flag that broadcasts the barge-in), keeping the
    #     default smart-turn STOP strategy. No mute state machine, so it's safe under steady.
    user_kwargs = {}
    if config.echo_guard:
        from pipecat.turns.user_mute import AlwaysUserMuteStrategy

        user_kwargs["user_mute_strategies"] = [AlwaysUserMuteStrategy()]
        logger.info("Echo-guard ON: mic muted while the bot speaks (half-duplex).")
    if not config.allow_interruptions:
        from pipecat.turns.user_start import (
            TranscriptionUserTurnStartStrategy,
            VADUserTurnStartStrategy,
        )
        from pipecat.turns.user_turn_strategies import UserTurnStrategies

        user_kwargs["user_turn_strategies"] = UserTurnStrategies(start=[
            VADUserTurnStartStrategy(enable_interruptions=False),
            TranscriptionUserTurnStartStrategy(enable_interruptions=False),
        ])
        logger.info("Interruptions OFF: the bot always finishes its turn (no barge-in).")
    user_params = None
    if user_kwargs:
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMUserAggregatorParams,
        )

        user_params = LLMUserAggregatorParams(**user_kwargs)
    aggregator = LLMContextAggregatorPair(context, user_params=user_params)

    pipeline = Pipeline([
        transport.input(),    # mic in (+ VAD set in transport params)
        stt,                  # speech -> text
        aggregator.user(),    # add user turn to context
        llm,                  # text -> streamed text
        tts,                  # text -> streamed audio
        avatar,               # audio -> lip-synced video+audio (server)
        meter,                # measure TTFO
        transport.output(),   # -> browser (audio + video)
        aggregator.assistant(),  # add bot turn to context
    ])

    _relax_bot_vad_stop_timeout()   # steady-mode screech fix (see the function's docstring)

    # Read-only transcript tap for the /nimbus/ chat bubbles (no pipeline structural change).
    _transcript = _TranscriptStore()
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[_make_transcript_observer(_transcript)],
    )
    global _active_task, _active_transcript
    _active_task = task   # let the /client/say endpoint inject typed turns into this task
    _active_transcript = _transcript   # served by /client/transcript for the chat bubbles

    async def _warmup_llm():
        # Open the HTTPS connection to the LLM now, so the TLS handshake is done
        # before the user's first message (kills cold start). OpenRouter is
        # OpenAI-compatible, so the chat.completions warmup applies. The weather chain
        # has no cheap warmup ping (and no _client), so skip it there -- this warmup
        # would crash on the custom service otherwise.
        if config.llm_provider == "weather_chain":
            return
        model = getattr(getattr(llm, "_settings", None), "model", None)
        try:
            await llm._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                stream=False,
            )
            logger.info("LLM connection pre-warmed.")
        except Exception as e:  # noqa: BLE001 — best-effort only
            logger.info(f"LLM warmup skipped: {e}")

    @transport.event_handler("on_client_connected")
    async def _on_connected(transport, client):
        logger.info("Client connected — warming LLM + sending greeting.")
        asyncio.create_task(_warmup_llm())   # warm the LLM in the background
        if config.is_thai:
            greeting = "สวัสดีค่ะ พร้อมแล้วค่ะ พูดได้เลย"
        elif config.is_mandarin:
            greeting = "嗨，我準備好了，請說。"
            # Personalize from memory if we already know the returning user.
            if memory is not None:
                hint = memory.greeting_hint()
                if hint:
                    greeting = "嗨，歡迎回來！" + hint  # "Hi, welcome back! " + hint
        else:
            greeting = "Hi, I'm ready — go ahead."
        # Speak a fixed greeting directly via TTS (no LLM round-trip needed).
        await task.queue_frames([TTSSpeakFrame(greeting)])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(transport, client):
        global _active_task, _active_connection
        # Only release EITHER global if it is still OURS -- a newer client may have already
        # claimed the slot (then this disconnect is us being kicked). The task global used to be
        # nulled unconditionally: a zombie session whose disconnect lands LATE (an ICE timeout
        # from a tab closed without a clean teardown) would then wipe the task of the LIVE session
        # that replaced it -> /client/say answers 409 "no active session"
        # while the avatar is visibly working.
        if _active_task is task:
            _active_task = None
        if _active_connection is conn:
            _active_connection = None
        logger.info(f"Client disconnected. TTFO summary: {meter.summary()}")
        if memory is not None:
            try:
                await memory.distill_and_save()  # grow the human's memory after the chat
            except Exception as e:  # noqa: BLE001
                logger.warning(f"memory distill skipped ({type(e).__name__})")
        await task.cancel()

    await PipelineRunner().run(task)


async def bot(runner_args: RunnerArguments) -> None:
    """Entrypoint the Pipecat dev runner calls with a configured transport."""
    from pipecat.runner.utils import create_transport

    # A/V SYNC MODE -- this picks the transport's video clock, and the two modes are
    # MUTUALLY EXCLUSIVE in pipecat 1.3.0 (verified in base_output.py):
    #   * sync_with_audio  -> a tagged OutputImageRawFrame is routed through the AUDIO
    #     queue and only displayed after its preceding audio (per-frame A/V pinning).
    #     The transport renders it via `_video_images`, which is ONLY read when
    #     video_out_is_live is FALSE (the non-live `_video_task_handler` branch).
    #   * video_out_is_live -> frames go through an INDEPENDENT timed video queue on
    #     their own wall-clock; `_video_images` (and thus every sync_with_audio frame)
    #     is NEVER read. So with is_live=True the whole sync_with_audio mechanism is a
    #     no-op -- video plays on a free-running clock and drifts vs the voice.
    # MuseTalk emits video_start/video_clock/video_end markers and its client
    # (musetalk_video.py) buffers the voice + tags frames sync_with_audio for true lip
    # pinning, so the transport MUST be NON-live for that to work. We couple them off the
    # same flag: sync on (steady) -> is_live False (real sync); sync off (live) -> is_live
    # True (legacy free-running clock, animates but drifts). Idle frames are pushed
    # untagged either way and animate via `_set_video_image`.
    sync_av = config.avatar_sync_with_audio

    # Split mode streams a fixed-size square mouth crop (config.avatar_split_size); the WebRTC
    # track MUST match those frame dimensions. Off = the full square portrait (avatar_size).
    _vout = config.avatar_split_size if config.avatar_split else config.avatar_size

    transport_params = {
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=True,
            # See the A/V SYNC MODE note above: non-live so sync_with_audio actually
            # pins each frame to its audio. (is_live would silently disable the sync.)
            video_out_is_live=not sync_av,
            # Square portrait; MUST equal the avatar server's MUSETALK_SIZE and the
            # service's image_size (config.avatar_size couples all three off MUSETALK_SIZE,
            # same discipline as MUSETALK_FPS below). Smaller = far less WAN bandwidth.
            video_out_width=_vout,
            video_out_height=_vout,
            # MUST equal the rate the avatar server PUSHES frames, or playout starves/
            # piles up and the face drifts behind the audio (the "laggy/desynced" drift,
            # then a freeze). The server pumps frames at config.avatar_fps (MuseTalk ~20 --
            # it frame-drops to that rate so a sub-realtime GPU stays realtime), so this
            # MUST track the same value. Coupled here (and in avatar.py) so they can never
            # diverge again.
            video_out_framerate=max(1, round(config.avatar_fps)),
            vad_analyzer=build_vad_params(),
        ),
    }
    # Single-connection policy: a fresh offer kicks the previous session BEFORE we build
    # this one, so the single-client avatar server (:8002) is released before the new
    # pipeline reaches for it (two live sessions fight over the one shared GPU).
    conn = getattr(runner_args, "webrtc_connection", None)
    global _active_connection
    old = _active_connection
    _active_connection = conn   # claim the slot first so the old session's disconnect handler won't clear it
    if old is not None and old is not conn:
        logger.info("New WebRTC offer -- disconnecting the previous session (single-connection policy).")
        try:
            await old.disconnect()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Kicking the previous session failed: {e!r}")

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, conn)


def _relax_bot_vad_stop_timeout() -> None:
    """THE steady-mode 'screech' root-cause fix (proven 2026-06-22).

    pipecat's output transport (`MediaSender._next_frame`) fires `_bot_stopped_speaking()` if no
    audio frame reaches its queue within `BOT_VAD_STOP_FALLBACK_SECS` (default **3s**) -- and that
    handler does `self._audio_buffer = bytearray()`, DISCARDING whatever partial audio is buffered.

    In `steady`/non-live sync the voice is held and released PACED TO RENDERED VIDEO frames. On a
    long reply the shared GPU render can stall > 3s, which starves the transport's audio queue, so
    the 3s timeout fires MID-TURN and discards the partial `_audio_buffer`. That discarded chunk is
    an arbitrary (usually ODD) byte count, so the remaining int16 PCM stream is left misaligned by
    an odd byte -> every subsequent sample straddles two real samples -> loud broadband noise to the
    end of the turn (the "screech"). Proven by byte-diffing the captures: a 1049-byte deletion at
    6.040s, speech otherwise bit-identical. `live` never hits this (it forwards audio continuously,
    so the queue is never starved 3s).

    We already drive an explicit `TTSStoppedFrame` per turn (`push_stop_frames=True` on the TTS
    service), which signals bot-stopped on its own, so the 3s audio-gap fallback is redundant here.
    Raise it so a render stall can never trigger the destructive discard -- a stall now just pauses
    the voice (steady's accepted behaviour) and it resumes CONTIGUOUS (clean), instead of screeching.

    The constant is read as a module global at `_next_frame()` call time (once per session, when the
    audio task starts), so patching the module attribute before the client connects takes effect.
    Knob: `BOT_VAD_STOP_FALLBACK_SECS` (seconds; <=0 leaves pipecat's 3s default)."""
    try:
        secs = float(os.getenv("BOT_VAD_STOP_FALLBACK_SECS", "600") or "600")
    except ValueError:
        secs = 600.0
    if secs <= 0:
        return
    try:
        from pipecat.transports import base_output
        base_output.BOT_VAD_STOP_FALLBACK_SECS = secs
        logger.info(f"BOT_VAD_STOP_FALLBACK_SECS -> {secs:g}s (steady-mode screech fix: a render "
                    f"stall can no longer discard the partial audio buffer mid-turn).")
    except Exception as e:  # noqa: BLE001 -- never block startup on this
        logger.warning(f"Could not relax BOT_VAD_STOP_FALLBACK_SECS: {e!r}")


def _configure_webrtc_video_bitrate() -> None:
    """Bound aiortc's VP8 send bitrate so the video stream FITS a remote/WAN link and
    can't starve it (the real cause of the "avatar trails the voice" stutter over the
    Thailand->Taiwan path).

    Why this is needed: pipecat's SmallWebRTCTransport hands raw frames to aiortc's VP8
    encoder, whose module-level limits are DEFAULT=500k, MIN=250k, MAX=1.5M (aiortc/codecs/
    vpx.py). It adapts DOWNWARD via REMB feedback, but (a) the 1.5M ceiling can overshoot a
    jittery consumer link -> packets queue -> the video falls progressively behind, and (b)
    the 250k floor can't absorb a worse dip -> loss -> freeze. pipecat's video_out_bitrate
    param is deprecated and wired to nothing, so the only place to set this is the aiortc
    module globals -- patched BEFORE the first encoder is created (it reads DEFAULT_BITRATE at
    init and the target_bitrate setter clamps to MIN/MAX). This keeps REMB's downward
    adaptation while capping the ceiling and lowering the floor (graceful degrade, no freeze).

    Knobs (bits/sec): WEBRTC_VIDEO_BITRATE (start point), _MAX (ceiling), _MIN (floor).
    Defaults suit a ~320px avatar over a multi-Mbps link; set _MAX=0 to leave aiortc as-is."""
    try:
        cap = int(os.getenv("WEBRTC_VIDEO_BITRATE_MAX", "600000") or "600000")
    except ValueError:
        cap = 600000
    if cap <= 0:
        return
    try:
        default = int(os.getenv("WEBRTC_VIDEO_BITRATE", "500000") or "500000")
        floor = int(os.getenv("WEBRTC_VIDEO_BITRATE_MIN", "120000") or "120000")
    except ValueError:
        default, floor = 500000, 120000
    try:
        from aiortc.codecs import vpx
    except Exception as e:  # noqa: BLE001
        logger.warning(f"WebRTC video bitrate config skipped (aiortc import: {e!r}).")
        return
    # Order matters: clamp default into the new [floor, cap] band.
    vpx.MIN_BITRATE = floor
    vpx.MAX_BITRATE = cap
    vpx.DEFAULT_BITRATE = max(floor, min(default, cap))
    logger.info(
        f"WebRTC VP8 bitrate bounded: min={floor} default={vpx.DEFAULT_BITRATE} max={cap} "
        f"(WEBRTC_VIDEO_BITRATE_MAX=0 to disable)."
    )


# All <head> patches for the served /client page collect here and ONE middleware injects
# them all. Why a shared list: each patch as its own middleware would race to serve the
# index (the outermost one wins and the others' patches silently vanish); a single
# serve-point keeps every env-gated patch additive and the prebuilt bundle untouched.
_client_head_patches: list[str] = []
_client_patch_middleware_installed = False
# Public-WebRTC ICE servers (STUN + TURN) served to the custom /nimbus/ client via
# GET /client/ice-config. Empty = no TURN configured -> public WebRTC off, tailnet behavior
# unchanged. Populated by _install_turn_ice_servers() from TURN_URLS/TURN_USERNAME/TURN_CREDENTIAL.
_ice_config_js: list[dict] = []
# When True, a fresh Cloudflare zero-signup TURN relay is appended per connection/request
# (see _cloudflare_turn). Set by _install_turn_ice_servers when TURN_CLOUDFLARE is enabled.
_cf_turn_enabled = False
# Set by run_bot so the /client/say endpoint (the /nimbus + /studio keyboard path) can inject a
# typed turn into the live pipeline. None between sessions.
_active_task = None

# Single-connection policy: the current live WebRTC connection. The avatar server is
# single-client and two sessions fight over the one shared GPU, so a new client kicks the
# previous one -- bot() disconnects this when a fresh offer arrives. None between sessions.
_active_connection = None

# Live conversation transcript for the custom /nimbus/ chat bubbles. The pipeline has no RTVI
# processor in this build, so instead of a data channel we tap frames with a READ-ONLY observer
# (no pipeline structural change) into a small ring buffer the client polls via /client/transcript.
# Set per session by run_bot; None between sessions.
_active_transcript = None


class _TranscriptStore:
    """Append-only ring buffer of {seq, role, text} the /client/transcript endpoint serves.

    role is 'user' (a committed STT transcription) or 'bot' (the assistant's aggregated reply text). seq lets the
    client poll incrementally (?since=N). Typed /say turns are echoed client-side already and never
    produce a TranscriptionFrame, so they are not double-added here.
    """

    def __init__(self, cap: int = 200):
        self._items: list[dict] = []
        self._seq = 0
        self._cap = cap
        # The in-progress user utterance (STT interim results). Not seq'd -- it's a single
        # slot the client renders as one live bubble that updates in place, then is cleared
        # when the finalized TranscriptionFrame commits. See /client/transcript.
        self._partial: dict | None = None

    def add(self, role: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        # KEEP this line (P45): it is the only log of each turn's committed reply/user text --
        # losing it is what hid the transcript corruption for weeks. Not temporary.
        logger.info(f"[commit-dbg] {role} {text!r}")
        self._seq += 1
        self._items.append({"seq": self._seq, "role": role, "text": text})
        if len(self._items) > self._cap:
            self._items = self._items[-self._cap:]

    def since(self, seq: int) -> list[dict]:
        return [it for it in self._items if it["seq"] > seq]

    def set_partial(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        self._partial = {"text": text, "updatedAt": time.time()}

    def clear_partial(self) -> None:
        self._partial = None

    @property
    def partial(self) -> dict | None:
        return self._partial


def _make_transcript_observer(store: "_TranscriptStore"):
    """A BaseObserver that records one user bubble per turn + the bot's aggregated reply text.

    Bot text arrives as a stream of LLMTextFrame tokens bracketed by LLMFullResponseStart/End;
    we accumulate between them and commit one 'bot' entry per reply. User STT arrives as
    InterimTranscriptionFrames (the live bubble) then one-or-more finalized TranscriptionFrames
    (one per speech pause); we accumulate the whole turn and commit ONE 'user' entry when the
    bot begins replying (LLMFullResponseStart). This only READS frames.
    """
    from pipecat.observers.base_observer import BaseObserver
    from pipecat.frames.frames import (
        TranscriptionFrame,
        InterimTranscriptionFrame,
        LLMTextFrame,
        LLMFullResponseStartFrame,
        LLMFullResponseEndFrame,
    )

    # No space between CJK segments (a space reads as a break mid-sentence); space for word langs.
    sep = "" if (config.is_mandarin or config.is_thai) else " "

    class _TranscriptObserver(BaseObserver):
        def __init__(self):
            super().__init__()
            self._buf = ""    # bot reply, accumulated between LLMFullResponseStart/End
            self._user = ""   # user turn, accumulated across STT segments until the bot replies
            # Dedupe: one frame OBJECT is pushed by several processors in turn, so the observer
            # sees it more than once and would append its text twice. Key on pipecat's own
            # `frame.id` (a monotonic per-frame counter) -- NEVER on `id(frame)`, which is the
            # MEMORY ADDRESS: CPython hands a freed frame's address straight back to the next
            # frame it allocates, so a brand-new token would hit this set and be silently DROPPED
            # from the bubble. That is exactly what happened (verified in pipeline.log: 自然語言處理
            # committed as 自言處理, 醫療保健 as 療保健 -- single chars deleted at random). The voice
            # was always fine (TTS reads the frames itself; this tap only feeds the chat bubbles),
            # which is why a corrupt transcript read as a mediocre LLM for weeks.
            self._seen: set[int] = set()

        async def on_push_frame(self, data):
            frame = data.frame
            fid = frame.id
            if fid in self._seen:
                return
            if isinstance(frame, LLMFullResponseStartFrame):
                # The bot starting to reply means the user's turn is complete: commit the WHOLE
                # accumulated turn as ONE bubble. Deepgram emits a TranscriptionFrame per speech
                # pause, so committing per-frame produced a bubble per pause ("a lot of bubbles").
                if self._user:
                    store.add("user", self._user)
                    self._user = ""
                store.clear_partial()  # the live bubble swaps for the committed one
                self._buf = ""
                self._seen.add(fid)
            elif isinstance(frame, LLMTextFrame):
                self._buf += frame.text or ""
                self._seen.add(fid)
            elif isinstance(frame, LLMFullResponseEndFrame):
                store.add("bot", self._buf)
                self._buf = ""
                store.clear_partial()  # backstop: drop any partial that never got a bot reply
                self._seen.add(fid)
                # Bound the dedupe set: frame ids are monotonic and never reused, so ids from a
                # FINISHED turn can never collide with a future frame -- keeping them would just
                # grow the set (one int per frame) for the whole process lifetime.
                self._seen = {fid}
            elif isinstance(frame, InterimTranscriptionFrame):
                # In-progress segment: show finalized-so-far + this live interim in the bubble.
                interim = (frame.text or "").strip()
                live = (self._user + sep + interim).strip() if self._user else interim
                store.set_partial(live)
                self._seen.add(fid)
            elif isinstance(frame, TranscriptionFrame):
                # A finalized STT segment -- accumulate; the single bubble commits at turn end
                # (LLMFullResponseStart above), not here.
                text = (frame.text or "").strip()
                if text:
                    self._user = (self._user + sep + text).strip() if self._user else text
                    store.set_partial(self._user)
                self._seen.add(fid)

    return _TranscriptObserver()


def _ensure_client_patch_middleware() -> bool:
    """Install (once) the middleware that serves /client with every registered head patch."""
    global _client_patch_middleware_installed
    if _client_patch_middleware_installed:
        return True
    try:
        from pathlib import Path as _Path

        import pipecat_ai_prebuilt
        from fastapi.responses import HTMLResponse

        from pipecat.runner.run import app
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Client page patch middleware skipped (import: {e!r}).")
        return False
    index_path = _Path(pipecat_ai_prebuilt.__file__).parent / "client" / "dist" / "index.html"
    if not index_path.is_file():
        logger.warning(f"Client page patch middleware skipped (no index.html at {index_path}).")
        return False

    @app.middleware("http")
    async def _inject_client_patches(request, call_next):
        # Nimbus transcript poll: the /nimbus/ chat polls this for new conversation lines (the bot's
        # spoken reply text + finalized user speech), captured by the read-only transcript observer.
        # ?since=<seq> returns only newer entries, plus "partial" = the in-progress user
        # utterance (STT interim, {"text","updatedAt"} | null) rendered as one live bubble.
        # JSON: {"items":[{"seq","role","text"}, ...], "partial": {...} | null}.
        if request.method == "GET" and request.url.path == "/client/transcript":
            import json as _json
            try:
                since = int(request.query_params.get("since", "0"))
            except (TypeError, ValueError):
                since = 0
            items = _active_transcript.since(since) if _active_transcript is not None else []
            partial = _active_transcript.partial if _active_transcript is not None else None
            return HTMLResponse(
                _json.dumps({"items": items, "partial": partial}),
                media_type="application/json",
            )
        # Nimbus split-mode overlay: proxy the avatar server's one-time compositing assets
        # (pristine background PNG + bbox) so /nimbus can paint the crisp background and
        # composite the streamed mouth crop over it. 404 when MUSETALK_SPLIT is off.
        if request.method == "GET" and request.url.path == "/client/avatar-overlay":
            import aiohttp
            url = config.avatar_url.rstrip("/") + "/overlay-assets"
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(url, timeout=aiohttp.ClientTimeout(total=5)) as r:
                        body = await r.read()
                        return HTMLResponse(body.decode("utf-8", "replace"),
                                            status_code=r.status,
                                            media_type="application/json")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[avatar-overlay] proxy failed: {e!r}")
                return HTMLResponse('{"split": false}', status_code=502,
                                    media_type="application/json")
        # Nimbus text send: inject a TYPED user turn (from the /nimbus/ chat box) into the live
        # pipeline as a real user message -> LLM -> TTS -> avatar speaks it. Voice-first stays the
        # primary path; this is the keyboard alternative and reuses the same _active_task inject as
        # the measure button. Body: {"text": "..."}.
        if request.method == "POST" and request.url.path == "/client/say":
            try:
                from log_setup import ensure_file_sink

                ensure_file_sink("pipeline")
                from pipecat.frames.frames import InterruptionFrame, LLMMessagesAppendFrame

                import json as _json
                raw = (await request.body())[:4000]
                text = (_json.loads(raw or b"{}").get("text") or "").strip()
                if not text:
                    return HTMLResponse("empty", status_code=400)
                if _active_task is None:
                    logger.warning("[say] no active session (client not connected?)")
                    return HTMLResponse("no active session", status_code=409)
                # A typed turn arriving mid-reply must BARGE IN like a spoken one. Voice barge-in
                # emits an InterruptionFrame (via the transport's broadcast_interruption); a typed
                # turn does NOT, so without this the current turn plays to completion and the new
                # answer only starts after it ends ("avatar runs until it ends"). InterruptionFrame
                # is a SystemFrame (out-of-band), so it is processed AHEAD of the DataFrame append:
                # it cancels the in-flight LLM/TTS, flushes the avatar, and commits the partial reply
                # -- THEN the append starts the new turn cleanly. Gated on allow_interruptions so the
                # shipped =0 "bot always finishes" mode still queues typed turns politely.
                frames = []
                if config.allow_interruptions:
                    frames.append(InterruptionFrame())
                frames.append(
                    LLMMessagesAppendFrame(messages=[{"role": "user", "content": text}], run_llm=True))
                await _active_task.queue_frames(frames)
                logger.info(f"[say] injected typed turn: {text!r}")
                return HTMLResponse("", status_code=204)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[say] failed: {e!r}")
                return HTMLResponse("error", status_code=500)
        # Client bootstrap config for the STATIC custom clients (/nimbus, /studio). They are
        # served as plain files, so they can't get a <head> RTCPeerConnection wrapper the way the
        # prebuilt /client page can -- they fetch this instead, before building the peer connection.
        # Carries:
        #   iceServers    -- STUN + TURN (empty over the tailnet path -> a default RTCPeerConnection).
        #   jitterBufferMs-- receive-side buffer target (CLIENT_JITTER_BUFFER_MS); absorbs WAN jitter.
        #   forceSpeaker  -- route the bot's voice to a phone's LOUDSPEAKER (CLIENT_FORCE_SPEAKER).
        # The last two used to exist ONLY as <head> injections into /client -- which MUSETALK_SPLIT=1
        # makes unsupported, so on the pages actually used they silently did nothing. Serving them
        # here is what makes those knobs real again for /nimbus + /studio.
        if request.method == "GET" and request.url.path == "/client/ice-config":
            import json as _json
            servers = list(_ice_config_js)
            if _cf_turn_enabled:  # append a FRESH Cloudflare relay (short-lived creds)
                cf_js, _ = _cloudflare_turn()
                if cf_js:
                    servers.append(cf_js)
            try:
                _jb = int(os.getenv("CLIENT_JITTER_BUFFER_MS", "0") or "0")
            except ValueError:
                _jb = 0
            _spk = (os.getenv("CLIENT_FORCE_SPEAKER", "1") or "1").lower() not in (
                "0", "false", "no", "off")
            return HTMLResponse(
                _json.dumps({"iceServers": servers, "jitterBufferMs": _jb, "forceSpeaker": _spk}),
                media_type="application/json",
                headers={"Cache-Control": "no-store"},
            )
        # Only the index page (exact /client or /client/); assets pass through to the mount.
        if request.method == "GET" and request.url.path in ("/client", "/client/"):
            try:
                html = index_path.read_text(encoding="utf-8").replace(
                    "<head>", "<head>" + "".join(_client_head_patches), 1
                )
                # no-store: a phone that cached the pre-patch index would silently miss
                # every injected fix (bit us 2026-07-04); the page is tiny, always refetch.
                return HTMLResponse(html, headers={"Cache-Control": "no-store"})
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Client page patch failed; serving default page: {e!r}")
        return await call_next(request)

    _client_patch_middleware_installed = True
    return True


def _restrict_ice_to_subnet() -> None:
    """Restrict WebRTC host candidates to ONE network (default: the Tailscale CGNAT range
    100.64.0.0/10), so ICE only ever offers the interface that can actually reach a remote
    tailnet viewer.

    Why this is needed (root cause of the intermittent mic, 2026-06-21): this box has
    several adapters -- Tailscale (100.x), Hyper-V (172.x), Radmin/Hamachi (26.x), LAN
    (192.168.x) -- and aiortc/aioice gather a host candidate for EVERY one. ICE then checks
    a large matrix of pairs, but only the Tailscale pair (100.x <-> the remote's 100.x) can
    actually reach a remote tailnet peer; the rest are dead. Worse, a marginal pair can win
    nomination and then drop ('Consent to send expired' in the logs) -> the audio track
    errors ('Media stream error; clearing track' / recv None) -> the mic dies mid-call. That
    is the "works sometimes, mostly not" symptom. The Tailscale pair is VERIFIED reachable in
    the logs (State.IN), so pinning ICE to it makes the stable path win immediately -- no
    relay/TURN needed.

    Patches aioice.ice.get_host_addresses (the host-candidate source) BEFORE the runner
    builds any peer connection, same module-global approach as the bitrate cap above. Safe by
    construction: WEBRTC_ICE_SUBNET=0 (or empty) disables it, and if the filter would drop
    EVERY address (e.g. Tailscale is down) it falls back to the full list so a local/LAN
    connection still works. A local browser can still reach the 100.x interface, so same-box
    testing is unaffected."""
    import ipaddress

    # We keep a SET of interfaces, not one. The classic tailnet pin (100.64/10) alone makes aiortc
    # derive its STUN srflx FROM the Tailscale interface, so a public visitor gets no reachable
    # candidate. But pinning to ONLY the internet-facing interface breaks the opposite clients:
    # a tailnet peer (or a device on the SAME home network, where reaching our public srflx needs
    # router hairpinning that home routers don't do) then has no usable pair and ICE never
    # completes. So in PUBLIC mode we advertise BOTH: the Tailscale interface (tailnet + same-LAN
    # clients pair on 100.x) AND the internet-facing default-route interface (its STUN srflx is the
    # public candidate a truly-external visitor reaches). We still DROP the noise (Hyper-V 172.x,
    # Radmin 26.x) so a marginal dead pair can't win nomination then drop ('Consent to send expired'
    # -> 'Media stream error; clearing track' -> audio dies + avatar video never renders).
    nets: list = []
    if (os.getenv("WEBRTC_PUBLIC", "") or "").strip().lower() in ("1", "true", "yes", "on"):
        import socket
        try:
            _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            _s.connect(("8.8.8.8", 80))  # no packet sent; just resolves the default-route source IP
            default_ip = _s.getsockname()[0]
            _s.close()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"WEBRTC_PUBLIC: default-route IP undetected ({e!r}); advertising all interfaces.")
            return
        # Tailscale range for tailnet/same-network clients + the internet-facing /32 for externals.
        tailnet = os.getenv("WEBRTC_ICE_SUBNET", "100.64.0.0/10") or "100.64.0.0/10"
        for spec in (tailnet, f"{default_ip}/32"):
            try:
                nets.append(ipaddress.ip_network(spec, strict=False))
            except ValueError:
                pass
        label = ", ".join(str(n) for n in nets)
        logger.info(f"WEBRTC_PUBLIC on: pinning ICE to {{{label}}} -- Tailscale (tailnet/same-LAN "
                    f"pairs) + internet-facing {default_ip} (public srflx); drops hyper-v/radmin noise.")
    else:
        subnet_str = os.getenv("WEBRTC_ICE_SUBNET", "100.64.0.0/10")
        if not subnet_str or subnet_str == "0":
            return
        try:
            nets.append(ipaddress.ip_network(subnet_str, strict=False))
        except ValueError:
            logger.warning(f"WEBRTC_ICE_SUBNET={subnet_str!r} invalid; ICE restriction skipped.")
            return
    try:
        from aioice import ice as _ice
    except Exception as e:  # noqa: BLE001
        logger.warning(f"ICE interface restriction skipped (aioice import: {e!r}).")
        return

    _orig = _ice.get_host_addresses

    def _filtered(use_ipv4: bool, use_ipv6: bool):
        addrs = _orig(use_ipv4, use_ipv6)
        kept = []
        for a in addrs:
            try:
                ip = ipaddress.ip_address(a)
            except ValueError:
                continue  # skip anything not a plain IP (e.g. scoped IPv6)
            if any(ip in n for n in nets):
                kept.append(a)
        if not kept:
            logger.warning(
                f"No host address in {[str(n) for n in nets]} (Tailscale down?); keeping all "
                f"{len(addrs)} addresses so the connection still works."
            )
            return addrs
        return kept

    _ice.get_host_addresses = _filtered
    logger.info(
        f"WebRTC ICE host candidates restricted to {[str(n) for n in nets]} "
        f"(was {len(_orig(True, False))} v4 addrs; WEBRTC_ICE_SUBNET=0 to disable)."
    )


_cf_turn_cache: dict = {"exp": 0.0, "js": None, "rtc": None}
_cf_refreshing = False   # a background refresh is in flight (don't stack them)


def _cloudflare_turn():
    """NON-BLOCKING accessor for the Cloudflare relay. Returns the cached (js, rtc) entry.

    Why this is not just `_cf_fetch()`: the fetch is a synchronous urlopen with an 8s timeout, and
    its two callers -- the GET /client/ice-config middleware and the patched
    SmallWebRTCConnection.__init__ (built while handling POST /api/offer) -- both run on THE SAME
    asyncio loop that carries aiortc's RTP media and the pipecat pipeline. Calling it there stalled
    the WHOLE LOOP: no packets out, no audio pumped -- a live call's voice and avatar freeze for up
    to 8s. Worse, a FAILED fetch cached nothing, so a dead endpoint was re-probed (and re-blocked)
    on every single request.

    So: never fetch on the loop. Serve the cache immediately (Cloudflare's creds outlive our 5-min
    cache window, so a slightly stale entry is still a working relay) and refresh in a THREAD when
    it ages out. A cold cache costs that ONE connection its relay (silent STUN-only fallback, the
    pre-existing failure mode) and self-heals for the next. Startup has no running loop, so the
    prime in _install_turn_ice_servers still fetches inline -- which is exactly where blocking is
    free."""
    global _cf_refreshing
    if time.time() >= _cf_turn_cache["exp"] and not _cf_refreshing:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return _cf_fetch()   # no loop yet (startup prime): blocking here is safe
        _cf_refreshing = True

        def _done(_fut):
            global _cf_refreshing
            _cf_refreshing = False

        loop.run_in_executor(None, _cf_fetch).add_done_callback(_done)
    return _cf_turn_cache["js"], _cf_turn_cache["rtc"]


def _cf_fetch():
    """Fetch FRESH Cloudflare TURN credentials with NO signup/account.

    BLOCKING (synchronous urlopen). Call it from a thread or before the loop starts -- never from
    the event loop itself; go through _cloudflare_turn() above.

    Cloudflare's speed test exposes a relay-credential endpoint (turn.cloudflare.com) that
    hands out short-lived username/credential pairs to any browser-style caller (a Referer
    header is all it wants). We reuse it as a zero-signup TURN relay so an off-tailnet visitor
    on a symmetric-NAT / UDP-restricted network can still reach the avatar (verified: yields a
    relay candidate on turn:3478 udp/tcp AND turns:5349 TLS, the firewall-proof one).

    Returns (js_entry: dict, rtc_entry: RTCIceServer) or (None, None) on any failure -> the
    caller silently degrades to STUN-only. Caches ~5 min (well under the credential TTL) so we
    don't hammer the endpoint. Best-effort by nature (undocumented endpoint); the drop-in
    upgrade is an official Cloudflare Realtime TURN key -> same turn.cloudflare.com servers,
    set TURN_URLS/TURN_USERNAME/TURN_CREDENTIAL instead and this path is bypassed."""
    now = time.time()
    try:
        import json as _json
        import urllib.request
        req = urllib.request.Request(
            "https://speed.cloudflare.com/turn-creds",
            headers={"Referer": "https://speed.cloudflare.com/", "User-Agent": "Mozilla/5.0"},
        )
        data = _json.loads(urllib.request.urlopen(req, timeout=8).read())
        urls = [u for u in data.get("urls", []) if u.startswith(("turn:", "turns:"))]
        user, cred = data.get("username"), data.get("credential")
        if not urls or not user or not cred:
            raise RuntimeError("endpoint returned no usable creds")
        from aiortc import RTCIceServer
        js = {"urls": urls, "username": user, "credential": cred}
        rtc = RTCIceServer(urls=urls, username=user, credential=cred)
        _cf_turn_cache.update(exp=now + 300, js=js, rtc=rtc)
        return js, rtc
    except Exception as e:  # noqa: BLE001
        # NEGATIVE cache: back off 30s. Without it a dead endpoint was re-fetched on EVERY
        # request. Keep any previously-good creds -- they outlive our cache window, so a stale
        # relay still beats no relay.
        _cf_turn_cache["exp"] = max(_cf_turn_cache["exp"], now + 30)
        logger.warning(f"Cloudflare TURN fetch failed ({e!r}); relay unavailable this attempt.")
        return _cf_turn_cache["js"], _cf_turn_cache["rtc"]


def _install_turn_ice_servers() -> None:
    """Enable PUBLIC WebRTC (a link a stranger can use) by advertising STUN + TURN.

    Why this is needed: over Tailscale Funnel (or any public tunnel) only the HTTPS page +
    /api/offer signaling is proxied -- the actual audio/video is UDP peer-to-peer over ICE and
    never flows through the tunnel. By default pipecat builds the server SmallWebRTCConnection
    with NO ice servers, so aiortc only gathers HOST candidates (LAN/Tailscale IPs a public
    browser can't reach). Behind CGNAT (common on TW ISPs) even STUN isn't enough, so a TURN
    relay is required for the media to traverse. This wires both sides to the same env-configured
    servers:
      * server (aiortc): monkeypatch SmallWebRTCConnection so every connection gets STUN+TURN ->
        it gathers srflx (public) + relay candidates the browser can reach.
      * client (prebuilt /client): the same <head> RTCPeerConnection wrapper pattern as the
        jitter buffer, injecting iceServers into every peer connection.
      * client (/nimbus): served statically, so it fetches GET /client/ice-config instead
        (see the middleware) -- fed by the _ice_config_js this function populates.

    GATE: active when TURN_URLS is set OR WEBRTC_PUBLIC=1. Off by default => no-op, the tailnet
    path is 100% unchanged (host candidates stay pinned by _restrict_ice_to_subnet; nothing new
    is advertised). WEBRTC_PUBLIC=1 alone advertises STUN only -- verified on this box the NAT is
    port-preserving (cone), so STUN-only reaches many public browsers with zero signup; add
    TURN_URLS as the fallback for a visitor whose NAT is symmetric. TURN creds are client-facing
    by design (the browser must use them), so embedding them in the served page / ice-config is
    expected. Env: WEBRTC_PUBLIC, TURN_URLS (comma-sep turn:/turns:), TURN_USERNAME,
    TURN_CREDENTIAL, STUN_URLS (default stun:stun.l.google.com:19302)."""
    global _ice_config_js, _cf_turn_enabled
    turn_urls = [u.strip() for u in (os.getenv("TURN_URLS", "") or "").split(",") if u.strip()]
    public = (os.getenv("WEBRTC_PUBLIC", "") or "").strip().lower() in ("1", "true", "yes", "on")
    if not turn_urls and not public:
        return  # public WebRTC disabled; tailnet behavior unchanged.
    # Cloudflare zero-signup TURN relay (see _cloudflare_turn): default ON when public and no
    # explicit TURN_URLS is set, so a symmetric-NAT / UDP-restricted visitor connects with no
    # account. TURN_CLOUDFLARE=0 opts out; =1 forces on even alongside a static TURN_URLS.
    cf_env = (os.getenv("TURN_CLOUDFLARE", "") or "").strip().lower()
    cf_turn = (cf_env in ("1", "true", "yes", "on")) or (
        public and not turn_urls and cf_env not in ("0", "false", "no", "off"))
    _cf_turn_enabled = cf_turn
    turn_user = (os.getenv("TURN_USERNAME", "") or "").strip()
    turn_cred = (os.getenv("TURN_CREDENTIAL", "") or "").strip()
    stun_urls = [u.strip() for u in
                 (os.getenv("STUN_URLS", "stun:stun.l.google.com:19302") or "").split(",")
                 if u.strip()]

    try:
        from aiortc import RTCIceServer
    except Exception as e:  # noqa: BLE001
        logger.warning(f"TURN ICE servers skipped (aiortc import: {e!r}).")
        return

    js: list[dict] = []
    rtc: list = []
    if stun_urls:
        js.append({"urls": stun_urls})
        rtc.append(RTCIceServer(urls=stun_urls))
    if turn_urls:
        turn_entry: dict = {"urls": turn_urls}
        if turn_user:
            turn_entry["username"] = turn_user
        if turn_cred:
            turn_entry["credential"] = turn_cred
        js.append(turn_entry)
        rtc.append(RTCIceServer(urls=turn_urls, username=turn_user or None,
                                credential=turn_cred or None))

    # Server side: inject the ice servers into every SmallWebRTCConnection (pipecat creates it
    # with ice_servers=None) so aiortc gathers srflx + relay candidates.
    try:
        from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
    except Exception as e:  # noqa: BLE001
        logger.warning(f"TURN ICE servers skipped (SmallWebRTCConnection import: {e!r}).")
        return
    if not getattr(SmallWebRTCConnection, "__turn_patched", False):
        _orig_init = SmallWebRTCConnection.__init__

        def _patched_init(self, ice_servers=None, *a, **k):
            if not ice_servers:
                servers = list(rtc)
                if cf_turn:  # append a FRESH Cloudflare relay per connection (creds are short-lived)
                    _cf_js, _cf_rtc = _cloudflare_turn()
                    if _cf_rtc is not None:
                        servers.append(_cf_rtc)
                ice_servers = servers
            return _orig_init(self, ice_servers, *a, **k)

        SmallWebRTCConnection.__init__ = _patched_init
        SmallWebRTCConnection.__turn_patched = True

    # Client side (prebuilt /client): wrap RTCPeerConnection to inject the same servers. Same
    # synchronous <head> pattern as the jitter buffer, so it runs before the ES-module bundle.
    _ice_config_js = js  # base; the /client/ice-config endpoint appends a FRESH Cloudflare relay
    # The prebuilt /client page embeds ICE statically (can't fetch async before the bundle builds
    # its peer connection), so bake a startup Cloudflare snapshot into it (best-effort; /studio +
    # /nimbus re-fetch fresh via /client/ice-config, so they never go stale).
    js_head = list(js)
    if cf_turn:
        _cf_js, _ = _cloudflare_turn()
        if _cf_js:
            js_head.append(_cf_js)
    import json as _json
    ice_json = _json.dumps(js_head)
    patch = (
        "<script>(()=>{const ICE=" + ice_json + ";const N=window.RTCPeerConnection;"
        "if(!N||N.__ice)return;const P=function(...a){const c=a[0]||{};"
        "if(!c.iceServers||!c.iceServers.length)c.iceServers=ICE;a[0]=c;return new N(...a);};"
        "P.prototype=N.prototype;P.__ice=1;window.RTCPeerConnection=P;"
        "console.log('[turn] '+ICE.length+' ICE server(s) injected');})();</script>"
    )
    if _ensure_client_patch_middleware():
        _client_head_patches.append(patch)
    if turn_urls:
        _relay = "STUN+TURN (static relay for symmetric NAT)"
    elif cf_turn:
        _relay = "STUN + Cloudflare zero-signup TURN relay (symmetric-NAT / UDP-restricted OK)"
    else:
        _relay = "STUN only (no TURN)"
    logger.info(
        f"Public WebRTC ICE servers ENABLED: {_relay} -> a public browser can reach the media. "
        f"Unset TURN_URLS/WEBRTC_PUBLIC to disable; TURN_CLOUDFLARE=0 to drop the Cloudflare relay."
    )


def _install_nimbus_client() -> None:
    """Serve the custom 'Nimbus AI' UI at /nimbus/ (the figma-to-code redesign).

    A self-contained vanilla-JS client (no build step) that speaks the SAME
    SmallWebRTC signaling as the prebuilt bundle -- POST /api/offer, then the
    avatar video + bot audio arrive as WebRTC tracks and the mic goes up the same
    connection. This is ADDITIVE: the prebuilt bundle at /client is untouched and
    stays the fallback. Mounted as StaticFiles so index.html + presenter.png serve
    from one dir; served no-store so a phone never caches a stale build.
    """
    from pathlib import Path as _Path

    client_dir = _Path(__file__).resolve().parent.parent / "local_services" / "nimbus_client"
    if not (client_dir / "index.html").is_file():
        logger.warning(f"Nimbus client not mounted (no index.html at {client_dir}).")
        return
    try:
        from starlette.staticfiles import StaticFiles
        from pipecat.runner.run import app
    except Exception as e:  # pragma: no cover - only when runner app isn't importable
        logger.warning(f"Nimbus client mount skipped ({e!r}).")
        return

    class _NoStoreStatic(StaticFiles):
        def is_not_modified(self, *a, **k):
            return False  # never 304 -> the phone always gets the latest build

        async def get_response(self, path, scope):
            resp = await super().get_response(path, scope)
            resp.headers["Cache-Control"] = "no-store"
            return resp

    app.mount("/nimbus", _NoStoreStatic(directory=str(client_dir), html=True), name="nimbus")
    logger.info("Nimbus UI mounted at /nimbus/ (custom client; /client prebuilt untouched).")

    # Sibling custom client for the "Leo" avatar preset (his face + cloned voice). Same
    # SmallWebRTC signaling + split-compositor; only the theme/branding differ. Additive --
    # whichever avatar preset is live streams to whichever page is open (one GPU, one avatar).
    studio_dir = _Path(__file__).resolve().parent.parent / "local_services" / "studio_client"
    if (studio_dir / "index.html").is_file():
        app.mount("/studio", _NoStoreStatic(directory=str(studio_dir), html=True), name="studio")
        logger.info("Studio UI mounted at /studio/ (Leo preset; /client + /nimbus untouched).")


if __name__ == "__main__":
    import sys

    # Windows consoles default to cp1252; Pipecat's runner prints emoji. Force
    # UTF-8 so startup doesn't crash on the banner.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # Durable per-process log at logs/pipeline.log (rotated, full tracebacks,
    # plus uvicorn/asyncio via the stdlib intercept). See log_setup.py.
    from log_setup import setup_logging

    setup_logging("pipeline")

    # Bound the VP8 send bitrate so the video fits a remote/WAN link (no starvation/freeze)
    # BEFORE any peer connection is built.
    #
    # (Removed 2026-07-14: six <head>-injection installers -- jitter buffer, phone-speaker route,
    # video-stall monitor, A/V-stats monitor, playout probe, measure button. They patched the
    # prebuilt /client page ONLY, and MUSETALK_SPLIT=1 makes /client unsupported, so on the pages
    # actually used (/nimbus, /studio) they were inert -- CLIENT_FORCE_SPEAKER=1 in .env was
    # loading nothing. The two that are real FEATURES (jitter buffer + speaker route) now live in
    # the static clients, fed by GET /client/ice-config; the other four were one-off diagnostics.)
    _configure_webrtc_video_bitrate()
    # Serve the custom 'Nimbus AI' redesign at /nimbus/ (additive; /client stays the fallback).
    _install_nimbus_client()
    # The /client/* API middleware (transcript, say, ice-config, avatar-overlay) is what makes
    # the nimbus/studio clients work -- it must NOT depend on the public-WebRTC gate. It used to
    # be installed only from inside _install_turn_ice_servers(), which returns early when
    # WEBRTC_PUBLIC=0 and no TURN_URLS -- so going tailnet-only silently 404'd the chat bubbles,
    # typed turns, AND the split-mode overlay (the avatar showed a raw mouth crop full-frame).
    # Idempotent: the TURN installer's own call becomes a no-op.
    _ensure_client_patch_middleware()
    # Pin ICE host candidates to the Tailscale interface so the stable 100.x<->100.x pair
    # wins immediately (kills the intermittent-mic ICE pollution -- see the function docstring).
    _restrict_ice_to_subnet()
    # PUBLIC link support: if TURN_URLS is set, advertise STUN+TURN so a stranger's browser
    # (behind CGNAT, over Funnel/a tunnel) can reach the media. No-op without TURN_URLS.
    _install_turn_ice_servers()
    from pipecat.runner.run import main

    main()
