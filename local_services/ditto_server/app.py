"""Ditto real-time lip-sync server (FastAPI + websocket).

Same wire contract as the MuseTalk server, so the pipeline client is identical:
  client -> server:
    text json {"type":"config","fps":25}
    text json {"type":"speech_start"} / {"type":"speech_end"} / {"type":"reset"}
    binary: 16-bit PCM mono @16 kHz audio chunks (TTS audio, resampled by client)
  server -> client:
    binary: raw RGB frame buffers (IMAGE_SIZE*IMAGE_SIZE*3 bytes) at `fps`
    text json: sync markers so the client paces audio to real rendered video --
      {"type":"video_start"}                : real lip-synced video starts
      {"type":"video_clock","frames":N}     : N real frames drained this turn
      {"type":"video_end"}                  : turn ended (flush trailing audio)

Implementation notes
--------------------
Drives Ditto's PyTorch path (antgroup/ditto-talkinghead, vendored under ./vendor)
on the local GPU. Two adaptations make it run here:

* **Frame interception.** Ditto's StreamSDK renders into a disk mp4 via its
  `writer_worker`. We subclass it and override that worker to push finished RGB
  frames onto a thread-safe queue instead, so we can stream them over the socket
  (mirrors how the MuseTalk server emits frames).
* **Continuous streaming.** We register the avatar once per connection and feed
  `run_chunk` across all utterances (online_mode, N_d=-1). The hubert audio
  windowing matches Ditto's own online example (6480-sample window, 3200 step,
  with a one-time pre-pad at stream start).

The Cython blend kernel was replaced with a NumPy equivalent in
vendor/.../core/utils/blend/__init__.py so no C compiler is needed.

Run (in the dedicated 'ditto' conda env):
    conda run -n ditto python -m local_services.ditto_server.app
"""
from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger

# --- paths -----------------------------------------------------------------
SERVER_DIR = Path(__file__).resolve().parent
VENDOR = SERVER_DIR / "vendor" / "ditto-talkinghead"
CKPT = VENDOR / "checkpoints"

# Ditto uses CWD-relative + package imports rooted at the repo; make it importable.
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

# onnxruntime-gpu needs CUDA 12 DLLs (cublasLt64_12.dll, cudnn) on the search
# path; without them its CUDA provider silently falls back to CPU and the avatar
# runs ~5x too slow. torch (cu128) already bundles those DLLs, so expose its lib
# dir BEFORE any onnxruntime import (which Ditto does lazily during setup()).
try:
    import torch as _torch

    os.add_dll_directory(str(Path(_torch.__file__).parent / "lib"))
    # TensorRT's nvinfer_10.dll lives in the tensorrt_libs package; expose it the
    # same way so the FP16 engines (DITTO_TRT) can load. Harmless if absent.
    try:
        import tensorrt_libs as _trt_libs

        os.add_dll_directory(str(Path(_trt_libs.__file__).parent))
    except Exception:  # noqa: BLE001
        pass
except Exception:  # noqa: BLE001
    pass

# AVATAR_REF is given relative to the project root; resolve before anything cd's.
AVATAR_REF = Path(os.getenv("AVATAR_REF", "assets/avatar.png")).resolve()

# --- knobs (env-overridable) ----------------------------------------------
IMAGE_SIZE = int(os.getenv("DITTO_SIZE", "512"))     # output frame is SIZE x SIZE
AUDIO_SR = 16000                                      # hubert expects 16 kHz
# Ditto's motion is generated at exactly 25 frames per audio-second (640 samples /
# frame @16kHz) -- this is fixed by the model and is the audio<->frame mapping.
NATIVE_FPS = 25
# OUTPUT fps. The client owns this (it sends it in the `config` message and paces
# the voice to it); this is only the pre-config fallback. When OUTPUT fps < 25 the
# server FRAME-DROPS to match (see _StridedQueue / frame_stride), so a sub-realtime
# GPU renders fewer frames and keeps up with the voice instead of drifting behind.
# Default 12 to match the client (stages/avatar.py) and the transport (main.py) -- ONE fps
# everywhere, so the frame-drop stride, the client's release clock, and the transport's
# video_out_framerate agree (a divergence drifts audio vs video). Also the basis for the
# LEAD_FRAMES cushion below, so it must equal the real operating fps.
DEFAULT_FPS = float(os.getenv("DITTO_FPS", "12"))
# Diffusion sampling steps -- the dominant render cost. Offline default is 50;
# online tolerates fewer. 25 is the known-good value; lowering it (e.g. 15) sped a
# single diffusion but, combined with the round-2 changes, slipped the render behind
# real-time -> use the new real_fps watchdog metric to tune this with evidence.
SAMPLING_TIMESTEPS = int(os.getenv("DITTO_STEPS", "25"))
# Diffusion granularity. The online worker renders once it has accumulated
# valid_clip_len = seq_frames(80) - overlap_v2 NEW audio features (~40ms each), running a
# full 80-frame denoise each pass. overlap_v2 ALSO sets how far the render's mouth motion
# LEADS the audio content (the model looks ahead within its window): the avatar_tune
# harness measured this lip lead and it scales with the window --
#   overlap 10 (vclip 70) -> mouth leads ~0.40s (corr 0.20)
#   overlap 45 (vclip 35) -> mouth leads ~0.18s (corr 0.29)
#   overlap 25 (vclip 55) -> mouth lead ~0.00s (corr 0.30) -- the CLEANEST, the default
# overlap 25 nails lip-sync with NO lead compensation needed. (Higher overlaps 60/70 sync
# erratically; see avatar_tuning_report.md.) The residual is the hardware wall: ~2.8s render
# latency per segment + a tail-vs-trim tradeoff, which no overlap escapes. seq_frames is
# model-bound (lmdm.seq_frames) -- only overlap_v2 is safe to change.
OVERLAP_V2 = int(os.getenv("DITTO_OVERLAP", "25"))
# Playout jitter buffer. Ditto produces frames in ~70-frame (2.8s) diffusion
# bursts at the edge of realtime, so the pump must build a lead buffer before it
# starts draining -- otherwise the per-clip compute gaps surface as stalls. Prime
# this many frames (25 ~= 1s @25fps) before playing; once playing, hold the last
# frame over short gaps and only fall back to the neutral portrait after the queue
# stays empty for IDLE_GRACE seconds (so inter-sentence gaps don't snap to neutral).
# Playout JITTER BUFFER, in OUTPUT frames -- the cushion that absorbs the diffuser's
# bursty render so playout stays steady. This is THE stability lever: the diffusion
# emits frames in ~bursts with compute gaps between; if the cushion is too small a gap
# drains the queue -> the pump holds the last frame (video freezes) while audio already
# released to the transport keeps playing -> the voice slides AHEAD of the frozen lips
# (the "unstable fps -> not sync" the user sees). A bigger cushion rides out the gaps so
# real frames never starve and the frame-clock keeps audio locked to video. Cost: the
# cushion is added start latency (we prime it before video_start). Default ~1.2s worth,
# computed from the output fps so it's right regardless of DITTO_FPS. Raise
# DITTO_LEAD_SECONDS if it still stutters; lower for snappier (riskier) starts.
LEAD_SECONDS = float(os.getenv("DITTO_LEAD_SECONDS", "0.5"))
LEAD_FRAMES = int(os.getenv("DITTO_LEAD_FRAMES", str(max(2, round(LEAD_SECONDS * DEFAULT_FPS)))))
IDLE_GRACE = float(os.getenv("DITTO_IDLE_GRACE", "0.8"))
# Idle-loop masking of the diffusion warmup. The felt "delay" is the ~2.2s first-frame
# delay (FFD = valid_clip_len of audio the diffuser must buffer before the first lip
# frame); on this GPU it can't be tuned away without breaking lip-sync (higher overlap /
# fewer steps cut FFD but destabilize the mouth -- measured). So instead of showing a
# FROZEN portrait while priming, we render a short idle clip ONCE at session start (feed
# silence -> the model produces a living face: blinks + micro-motion, measured ~4.6 eye
# motion w/ blink spikes) and LOOP it during priming/idle. The warmup then reads as
# natural listening, not lag. 0 disables (back to the static neutral frame). Default
# renders ~one diffusion window of idle (valid_clip_len/5 + 1 chunks). Negative disables
# (back to the static neutral frame); >0 sets an explicit chunk count.
IDLE_CAPTURE_CHUNKS = int(os.getenv("DITTO_IDLE_CAPTURE_CHUNKS", "0"))  # 0 = auto, <0 = off
# End-of-sentence flush size, in run_chunk calls (5 features each). At speech_end we feed
# silence so the worker diffuses the real trailing audio it still holds. A FULL window
# (valid_clip_len//5 + 1) guarantees no truncation -- but in LIVE mode (no tail-trim) every
# silence frame it renders PLAYS as a long closed-mouth tail (overlap 25 -> ~4.4s of it).
# A small flush (e.g. 3 = ~0.6s) pushes out most of the real tail without the long silent
# segment; the very last fraction renders when the next input arrives (fine in a continuous
# reply). 0 = auto = the full safe window (the compute-first default, where the trim hides it).
FLUSH_CHUNKS = int(os.getenv("DITTO_FLUSH_CHUNKS", "0"))
# Compute-first playback (default ON, COMBINED with frame-drop). It waits until a sentence
# is FULLY rendered before playing it, then drains it with the voice frame-clocked to the
# real frames -> the sentence can't lag at the start (video_start and the voice release
# together). Frame-dropping (DITTO_FPS=12.5) halves the frames so the pre-render is fast
# enough to sustain. Cost: a ~3-4s render-wait before each short sentence shows. Set 0 for
# legacy live-streaming (lower latency, but lips lag the voice at the start of each line).
# LIVE streaming is the default: the GPU sustains the output fps (DITTO_FPS=8 at overlap 25
# -> measured real_fps 8.0-8.5, frame_q never starves -> NO freeze), so there's no need to
# pre-render. Live also never truncates (no tail-trim/queue-clear -- the bug that cut
# sentences in compute-first when the next sentence rendered into the queue early). The voice
# is frame-clocked to real frames, so it stays synced. Set 1 only if you raise DITTO_FPS above
# what the GPU sustains (then pre-render trades a per-sentence wait to avoid starvation).
PRERENDER = (os.getenv("DITTO_PRERENDER", "0") or "0").lower() in ("1", "true", "yes", "on")
PRERENDER_STABLE_MS = int(os.getenv("DITTO_PRERENDER_STABLE_MS", "500"))   # queue-idle = done
PRERENDER_MAX_FRAMES = int(os.getenv("DITTO_PRERENDER_MAX_FRAMES", "250"))  # ~10s cap
PRERENDER_MAX_WAIT_S = float(os.getenv("DITTO_PRERENDER_MAX_WAIT_S", "12"))  # safety
DATA_ROOT = str(CKPT / "ditto_pytorch")
CFG_PKL = str(CKPT / "ditto_cfg" / "v0.4_hubert_cfg_pytorch.pkl")

# TensorRT FP16 acceleration (the hybrid). The per-frame bottleneck on this GPU is
# the photoreal render, dominated by the decoder (~59 ms at fp32 -> ~14 ms as an
# FP16 TensorRT engine, measured 4.3x on this sm_120 card). DITTO_TRT swaps the
# heavy, plugin-free models (decoder/stitch/appearance) for Blackwell FP16 engines
# built by build_trt.py, while warp_network (custom GridSample3D plugin, Linux-only)
# and everything else STAY on the PyTorch path. Net: ~8 fps -> realtime, same visual
# (each engine is numerically validated vs fp32 before it's accepted). Set DITTO_TRT=0
# to run the original all-PyTorch path (fallback). Engines are GPU-arch-specific, so
# the build is one-time on this card; if they're missing we log and fall back.
USE_TRT = (os.getenv("DITTO_TRT", "1") or "1").lower() in ("1", "true", "yes", "on")
TRT_DIR = CKPT / "ditto_trt_blackwell"
# Which models get swapped to engines, and the cfg section (under base_cfg) for each.
TRT_SWAPS = {
    "decoder_cfg": "decoder_fp16.engine",
    "stitch_network_cfg": "stitch_network_fp16.engine",
    "appearance_extractor_cfg": "appearance_extractor_fp16.engine",
    # warp_network needs the custom GridSample3D op, which requires a TRT plugin.
    # We built that plugin for sm_120 (plugin_build/), so warp can now be an engine
    # too -- the biggest remaining per-frame GPU cost (~47ms PyTorch -> FP16 engine,
    # validated 0.5% mean err vs PyTorch). Gated on the plugin loading in _enable_trt:
    # if the .dll is absent/fails, warp is dropped from the swaps and stays PyTorch.
    "warp_network_cfg": "warp_network_fp16.engine",
}
# The custom plugin .dll (registers "GridSample3D"), staged next to the engines.
WARP_PLUGIN = TRT_DIR / "grid_sample_3d_plugin.dll"
# NOTE on the diffusion net (LMDM): we measured it on TensorRT too, but it gave NO
# net end-to-end gain. Its TRT path runs the 25-step sampling loop in numpy
# (core/models/lmdm.py `_call_np`), so each step pays a host<->device roundtrip that
# cancels the compute saving; and the true render ceiling here is the SERIAL per-frame
# pipeline (warp+decode+stitch+putback ~62ms/frame, GIL-bound), not the diffusion. So
# the LMDM swap is deliberately NOT wired in. build_trt.py can still produce the engine
# for experimentation. warp (the other big per-frame chunk) IS now a TRT engine -- it
# needs the custom GridSample3D plugin, which we built for sm_120 (see _load_grid_sample_plugin).


def _load_grid_sample_plugin() -> bool:
    """Register the custom GridSample3D TRT plugin so the warp engine can deserialize.

    The warp ONNX uses a 5D grid_sample (op `GridSample3D`) that TensorRT has no
    built-in for; the engine therefore needs our plugin DLL loaded process-wide
    BEFORE any deserialize. Best-effort: returns False if the DLL is missing or
    fails to register (then warp is dropped from the swaps and stays PyTorch)."""
    import ctypes

    if not WARP_PLUGIN.is_file():
        logger.warning(f"GridSample3D plugin not found at {WARP_PLUGIN}; warp stays PyTorch.")
        return False
    try:
        ctypes.CDLL(str(WARP_PLUGIN), mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        import tensorrt as trt

        trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"GridSample3D plugin load failed ({e}); warp stays PyTorch.")
        return False


def _enable_trt() -> str | None:
    """Wire the FP16 engines in and return the hybrid cfg path (or None to fall back).

    Two steps, both contained to startup:
    1. Make Ditto's `load_model` use our torch-native TRT runner for `.engine` files.
       Its lazy `from .tensorrt_utils import TRTWrapper` would otherwise import the
       vendored module, which needs `cuda-python` (absent) and the TRT-8 Linux ABI.
       We pre-seed sys.modules with a tiny stub exposing TRTWrapper=TRTRunner, so the
       broken import never runs.
    2. Generate a hybrid cfg pkl: the all-PyTorch cfg with the swapped models'
       `model_path` repointed to absolute engine paths. parse_cfg's `_check_path`
       returns an absolute existing file as-is, so load_model sees `.engine` and
       dispatches to TensorRT for exactly those three; everything else stays PyTorch.
    """
    import pickle
    import types

    swaps = dict(TRT_SWAPS)
    # warp needs the GridSample3D plugin; register it (best-effort). If it can't load,
    # drop warp so the rest of the hybrid still works (warp falls back to PyTorch).
    warp_on = _load_grid_sample_plugin()
    if not warp_on:
        swaps.pop("warp_network_cfg", None)

    missing = [n for n in swaps.values() if not (TRT_DIR / n).is_file()]
    # A missing warp engine is non-fatal (drop it); a missing CORE engine means the
    # hybrid isn't built -> fall back to all-PyTorch.
    if "warp_network_fp16.engine" in missing:
        swaps.pop("warp_network_cfg", None)
        warp_on = False
        missing.remove("warp_network_fp16.engine")
    if missing:
        logger.warning(
            f"DITTO_TRT on but engines missing: {missing}. Run "
            f"`python -m local_services.ditto_server.build_trt`. Falling back to PyTorch."
        )
        return None

    from local_services.ditto_server.trt_runner import TRTRunner

    stub = types.ModuleType("core.utils.tensorrt_utils")
    stub.TRTWrapper = TRTRunner
    sys.modules["core.utils.tensorrt_utils"] = stub

    with open(CFG_PKL, "rb") as f:
        cfg = pickle.load(f)
    for section, engine_name in swaps.items():
        cfg["base_cfg"][section]["model_path"] = str(TRT_DIR / engine_name)
    hybrid = CKPT / "ditto_cfg" / "_hybrid_blackwell_cfg.pkl"
    with open(hybrid, "wb") as f:
        pickle.dump(cfg, f)
    warp_msg = "warp" if warp_on else "warp(PyTorch)"
    logger.info(
        f"DITTO_TRT on: decoder/stitch/appearance/{warp_msg} -> Blackwell FP16 engines; "
        "diffusion + aux models stay PyTorch."
    )
    return str(hybrid)

# hubert online windowing (from Ditto's inference.py online branch)
CHUNKSIZE = (3, 5, 2)
STEP = CHUNKSIZE[1] * 640                              # 3200 samples advance
SPLIT_LEN = int(sum(CHUNKSIZE) * 0.04 * AUDIO_SR) + 80  # 6480 sample window
PREPAD = CHUNKSIZE[0] * 640                            # 1920 zero pre-pad at start

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Ditto's models once at startup (the FastAPI on_event hooks are
    # deprecated; lifespan is the supported replacement). `engine` is defined
    # below and resolved at call time, so the forward reference is fine.
    engine.load()
    yield


app = FastAPI(title="Ditto realtime", lifespan=lifespan)

# One GPU, shared models -> single inference at a time across the process.
_render_lock = asyncio.Lock()
# The SDK (and its worker threads) is shared and single-client by design. Hold
# this for a whole connection so a reconnect can't re-run setup() while the prior
# session's threads are still alive (which corrupts the pipeline -> frozen avatar).
_session_lock = asyncio.Lock()


class _StridedQueue(queue.Queue):
    """A Queue that keeps only every `stride`-th real item, dropping the rest.

    This is the avatar's fps lever. It sits on the motion->render handoff, BEFORE
    the expensive per-frame warp/decode/putback stages, so a dropped frame is never
    rendered (the GPU does proportionally less work). At stride 2 the renderer only
    has to produce ~12.5 fps instead of 25 -- which a sub-realtime GPU can sustain in
    real time, so the video keeps up with the voice instead of drifting behind. The
    motion is relative to a FIXED d0 (motion_stitch._mix_s_d_info), so subsampling
    just yields evenly-spaced frames (choppier, but correct) -- no drift accumulates.

    `None` sentinels (end-of-stream) always pass. `stride` is read live via
    stride_getter so it can change when the client sends its fps config."""

    def __init__(self, maxsize=0, *, stride_getter):
        super().__init__(maxsize)
        self._stride_getter = stride_getter
        self._n = 0

    def put(self, item, block=True, timeout=None):
        if item is None:
            return super().put(item, block, timeout)
        stride = max(1, int(self._stride_getter()))
        keep = (self._n % stride == 0)
        self._n += 1
        if keep:
            return super().put(item, block, timeout)
        return None  # dropped: never rendered (the whole point)


def _make_sdk_class():
    """Subclass Ditto's online StreamSDK to divert rendered frames to a queue."""
    from stream_pipeline_online import StreamSDK

    class StreamingSDK(StreamSDK):
        frame_stride = 1  # 1 = full 25 fps; 2 = render every other frame (~12.5 fps)

        def attach_sink(self, frame_queue: "queue.Queue"):
            self._frame_sink = frame_queue

        def setup(self, *args, **kwargs):
            super().setup(*args, **kwargs)
            # Swap the motion->render handoff queue for the frame-dropping one, so we
            # can shed frames BEFORE the costly warp/decode/putback workers. Safe to
            # swap here: setup() has only just started the workers and no audio has
            # been fed yet, so both queues are empty and nothing is in flight.
            self.motion_stitch_queue = _StridedQueue(
                self.motion_stitch_queue.maxsize,
                stride_getter=lambda: self.frame_stride,
            )

        def _writer_worker(self):
            # Override: push finished RGB frames to our queue instead of disk.
            try:
                while not self.stop_event.is_set():
                    try:
                        item = self.writer_queue.get(timeout=1)
                    except queue.Empty:
                        continue
                    if item is None:
                        break
                    self._frame_sink.put(item)  # res_frame_rgb (H,W,3 uint8 RGB)
            except Exception as e:  # noqa: BLE001
                self.worker_exception = e
                self.stop_event.set()

    return StreamingSDK


class DittoEngine:
    """Loads Ditto models once; registers the avatar per connection."""

    def __init__(self):
        self._sdk_cls = None
        self.sdk = None
        self._neutral = None  # source portrait, used as the idle/neutral frame
        # Idle loop (living resting face) for warmup masking, captured ONCE on the first
        # session and reused: the avatar portrait is identical every connection, so the
        # idle motion is too -- no need to pay the ~one-diffusion render cost per connect.
        self.idle_frames: list[bytes] = []

    def load(self):
        if self._sdk_cls is None:
            if not Path(DATA_ROOT).exists():
                raise RuntimeError(f"Ditto checkpoints not found at {DATA_ROOT}")
            # The offline smoke ran with cwd inside the repo; match it so any
            # CWD-relative model path inside Ditto resolves the same way.
            os.chdir(VENDOR)
            self._sdk_cls = _make_sdk_class()
            cfg_pkl = CFG_PKL
            if USE_TRT:
                cfg_pkl = _enable_trt() or CFG_PKL
            backend = "TensorRT-FP16 hybrid" if cfg_pkl != CFG_PKL else "pytorch"
            logger.info(f"Loading Ditto ({backend}) from {DATA_ROOT} ...")
            self.sdk = self._sdk_cls(cfg_pkl, DATA_ROOT)
            logger.info("Ditto models loaded.")
            # Prebuild a neutral frame from the source portrait.
            import cv2

            img = cv2.imread(str(AVATAR_REF), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Avatar reference not found: {AVATAR_REF}")
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            self._neutral = _frame_to_bytes(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def neutral_frame(self) -> bytes:
        return self._neutral


def _frame_to_bytes(frame_rgb: np.ndarray) -> bytes:
    import cv2

    if frame_rgb.shape[:2] != (IMAGE_SIZE, IMAGE_SIZE):
        frame_rgb = cv2.resize(frame_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    return np.ascontiguousarray(frame_rgb, dtype=np.uint8).tobytes()


engine = DittoEngine()

# --- live status for the pipeline's debug dashboard ------------------------
# Updated by the per-session watchdog and read by GET /status. Read-only: this
# adds visibility, it does NOT change rendering. The dashboard maps these to the
# documented avatar failure modes (CPU fallback, stalled render, worker crash).
_STATUS: dict = {
    "session_active": False,
    "fps": 0.0,          # realized output fps (sent delta / interval)
    "fps_target": DEFAULT_FPS,
    "queue": 0,
    "speaking": False,
    "workers_stopped": False,
    "sent": 0,
}
_session_refs: dict | None = None  # live handles for the active session, or None

# --- per-stage render profiling (the optimization scoreboard) --------------
# Each per-frame render stage runs in its own worker thread; we wrap the stage
# callables (warp/decode/stitch/putback) with a timer so the watchdog can log
# where the ~62ms/frame goes. Wall-clock timing is meaningful because each stage
# currently SYNCs the GPU inside (.cpu().numpy()), so the time includes compute.
# Single writer per stage (one thread each) -> the lock only guards the watchdog
# read+reset. Near-free. Default OFF (DITTO_PROFILE=1 to enable while tuning): the
# wrappers replace per-frame stage attributes on the SHARED sdk, so they must be
# (a) idempotent across sessions and (b) only applied to PURE callables -- never
# motion_stitch, whose .setup()/.d0/.set_Nd are used by the SDK's setup().
import threading as _threading

PROFILE = (os.getenv("DITTO_PROFILE", "0") or "0").lower() in ("1", "true", "yes", "on")
_STAGE_LOCK = _threading.Lock()
_STAGE_MS: dict = {}  # name -> [count, total_ms, max_ms]


def _timed(name: str, fn):
    """Wrap a stage callable to accumulate per-frame timing under `name`."""
    def wrapped(*a, **k):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        dt = (time.perf_counter() - t0) * 1000.0
        with _STAGE_LOCK:
            c = _STAGE_MS.get(name)
            if c is None:
                _STAGE_MS[name] = [1, dt, dt]
            else:
                c[0] += 1
                c[1] += dt
                c[2] = max(c[2], dt)
        return out
    wrapped._profiled = True  # idempotency marker (the sdk is shared across sessions)
    return wrapped


def _profile_attach(sdk) -> None:
    """Wrap the per-frame stage callables in place so the workers time them.

    The workers call ``self.warp_f3d(...)`` etc. each iteration, so replacing the
    instance attributes is picked up with no edit to the vendored worker code. ONLY
    pure callables (WarpF3D/DecodeF3D/PutBack __call__) are wrapped -- NOT
    motion_stitch (the SDK calls motion_stitch.setup()/.d0, which a function lacks).
    Idempotent: the sdk is shared, so skip anything already wrapped."""
    if not PROFILE:
        return
    for attr, name in (("warp_f3d", "warp"), ("decode_f3d", "decode"), ("putback", "putback")):
        fn = getattr(sdk, attr, None)
        if fn is not None and not getattr(fn, "_profiled", False):
            setattr(sdk, attr, _timed(name, fn))


def _profile_drain() -> str:
    """Return a compact 'stage avg/max ms' summary and reset the counters."""
    if not PROFILE:
        return ""
    with _STAGE_LOCK:
        parts = []
        for name in ("warp", "decode", "putback"):
            c = _STAGE_MS.get(name)
            if c and c[0]:
                parts.append(f"{name}={c[1] / c[0]:.1f}/{c[2]:.0f}")
        _STAGE_MS.clear()
    return " ".join(parts)


def _gpu_stats() -> dict:
    """GPU utilization + VRAM + temp via nvidia-smi (pynvml isn't installed).

    Cheap enough to call once per 2s watchdog tick. Returns zeros if nvidia-smi
    isn't on PATH. Run inside asyncio.to_thread so the subprocess never blocks the
    event loop."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip().splitlines()[0]
        util, used, total, temp = (int(x.strip()) for x in out.split(","))
        return {"gpu_util": util, "vram_used": used, "vram_total": total, "gpu_temp": temp}
    except Exception:  # noqa: BLE001 -- nvidia-smi missing / parse error
        return {"gpu_util": None, "vram_used": None, "vram_total": None, "gpu_temp": None}


def _cuda_flags() -> dict:
    """Best-effort GPU check. onnx_cuda False after a session has loaded ==
    the silent CPU-fallback trap (avatar ~5x too slow)."""
    flags = {"torch_cuda": None, "onnx_cuda": None}
    try:
        flags["torch_cuda"] = bool(_torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        pass
    try:
        import onnxruntime  # imported lazily by Ditto during setup()

        flags["onnx_cuda"] = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:  # noqa: BLE001
        pass
    return flags


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    fps = DEFAULT_FPS
    closed = asyncio.Event()
    speaking = asyncio.Event()
    loop = asyncio.get_running_loop()

    frame_q: "queue.Queue" = queue.Queue(maxsize=600)   # rendered RGB frames
    audio_buf = np.zeros(0, dtype=np.float32)
    started = False  # whether we've pre-padded the stream

    # Serialize whole sessions: block a reconnect until the prior one has fully
    # torn down its worker threads (else re-running setup() corrupts the SDK).
    await _session_lock.acquire()
    logger.info("New session: registering avatar...")

    # --- register avatar + start the streaming session (once per connection) ---
    sdk = engine.sdk
    sdk.attach_sink(frame_q)
    async with _render_lock:
        await asyncio.to_thread(
            sdk.setup,
            str(AVATAR_REF),
            str(SERVER_DIR / "_discard.mp4"),
            online_mode=True,
            sampling_timesteps=SAMPLING_TIMESTEPS,
            overlap_v2=OVERLAP_V2,   # smaller valid_clip_len -> lower render latency
            N_d=-1,
        )
    sdk.setup_Nd(N_d=-1)
    # Apply the default-fps frame-drop stride NOW (before warmup/idle capture), not just
    # when the client's `config` arrives -- otherwise the pre-config idle render runs at
    # full 25 fps (stride 1), doubling its cost. The client re-confirms the same stride.
    sdk.frame_stride = max(1, round(NATIVE_FPS / DEFAULT_FPS)) if DEFAULT_FPS > 0 else 1
    _profile_attach(sdk)  # wrap per-frame stages for timing (no-op if DITTO_PROFILE=0)
    # How many NEW audio features the worker needs before each diffusion fires.
    # Drives the warmup + per-turn flush below. (~40ms / feature; run_chunk = 5.)
    valid_clip_len = int(getattr(sdk.audio2motion, "valid_clip_len", 80 - OVERLAP_V2))
    logger.info(f"Avatar registered; streaming session ready (valid_clip_len={valid_clip_len}).")

    sent = 0  # frames pushed to client (for the watchdog log)
    # Instrumentation, read + reset by the watchdog every 2s.
    render_n = 0            # run_chunk (diffusion) calls this interval
    render_ms_total = 0.0
    render_ms_max = 0.0
    q_peak = 0             # max frame_q depth the pump saw this interval
    real_cum = 0           # cumulative REAL frames drained (true render throughput)
    seg_samples = 0        # audio samples fed for the CURRENT sentence (for prerender:
                           # expected frames = seg_samples / 640 @16kHz/25fps)
    feat_fed = 0           # cumulative audio features fed since warmup (5 per run_chunk);
                           # tracks the worker's diffusion boundary for the minimal flush

    async def _run_chunk(chunk: np.ndarray):
        """Feed one SPLIT_LEN window to the renderer (serialized on the GPU). Times
        the hubert feature-extraction (the diffusion itself runs in the worker thread)."""
        nonlocal render_n, render_ms_total, render_ms_max, feat_fed
        t0 = time.perf_counter()
        async with _render_lock:
            await asyncio.to_thread(sdk.run_chunk, chunk, CHUNKSIZE)
        dt = (time.perf_counter() - t0) * 1000.0
        render_n += 1
        render_ms_total += dt
        render_ms_max = max(render_ms_max, dt)
        feat_fed += 5  # wav2feat yields 5 features per SPLIT_LEN chunk

    # Warm up the renderer NOW. The online worker's first diffusion block only
    # initializes d0 and emits zero frames, so push silence through it before any
    # real audio -- otherwise the user's very first turn is eaten by d0 init and
    # renders nothing. Discard whatever (if anything) it produced.
    _silence = np.zeros(SPLIT_LEN, dtype=np.float32)
    for _ in range(valid_clip_len // 5 + 2):
        await _run_chunk(_silence)
    await asyncio.sleep(0.3)  # let the worker thread settle
    try:
        while True:
            frame_q.get_nowait()
    except queue.Empty:
        pass
    feat_fed = 0  # align the flush counter to the worker's boundary after d0 init
    logger.info("Renderer warmed (d0 initialized).")

    # Capture a short IDLE loop to mask the warmup (see DITTO_IDLE_CAPTURE_CHUNKS). d0 is
    # initialized now, so feeding a bit more silence renders the resting face WITH its
    # natural blinks/micro-motion. We keep those frames and loop them during priming/idle
    # instead of a frozen portrait. This is the only place the frame_q is ours alone (the
    # pump task starts below), so draining it here is safe. Captured ONCE per server life
    # and cached on the engine (same portrait every connection), so only the first session
    # pays the render cost; later connects reuse it and start streaming immediately.
    idle_frames: list[bytes] = engine.idle_frames
    if IDLE_CAPTURE_CHUNKS >= 0 and not idle_frames:
        n_idle = IDLE_CAPTURE_CHUNKS if IDLE_CAPTURE_CHUNKS > 0 else (valid_clip_len // 5 + 1)
        for _ in range(n_idle):
            await _run_chunk(_silence)
        # CRITICAL: keep draining until the worker fully quiesces. The serial render
        # pipeline produces idle frames for SECONDS after the feed stops; if we stop
        # draining early, the stragglers stay in frame_q and leak into the first real
        # turn -- corrupting its markers/coverage (measured: s0 coverage 0.25). So poll
        # repeatedly, collecting everything, and only stop after TWO consecutive empty
        # polls once we have frames -> frame_q is guaranteed clean for the first turn.
        deadline = loop.time() + 10.0
        empties = 0
        while loop.time() < deadline:
            await asyncio.sleep(0.25)
            got = 0
            try:
                while True:
                    idle_frames.append(_frame_to_bytes(frame_q.get_nowait()))
                    got += 1
            except queue.Empty:
                pass
            if got == 0 and idle_frames:
                empties += 1
                if empties >= 2:       # quiesced: render done and queue stays empty
                    break
            else:
                empties = 0
        feat_fed = 0  # realign the flush counter after the extra idle feed
        logger.info(f"Idle loop captured ({len(idle_frames)} frames) for warmup masking.")
    elif idle_frames:
        logger.info(f"Idle loop reused ({len(idle_frames)} cached frames); no capture cost.")

    # Publish live handles so GET /status reflects this session in real time.
    global _session_refs
    _session_refs = {"frame_q": frame_q, "speaking": speaking, "sdk": sdk}
    _STATUS.update({"session_active": True, "fps_target": fps, "workers_stopped": False})

    async def pump():
        """Emit a steady `fps` RGB stream through a jitter buffer so the bursty,
        edge-of-realtime renderer plays back continuously.

        State machine, driven by the queue (not the per-sentence `speaking` flag):
          idle/priming -> show neutral, DON'T drain, until LEAD_FRAMES are buffered
          playing      -> drain one frame/tick; on underflow hold the last frame,
                          and only return to neutral after IDLE_GRACE of empty queue
        Holding the last frame over short gaps is what keeps motion continuous
        across the per-clip diffusion gaps and inter-sentence pauses.

        It is also the **sync clock** for the client: alongside the binary frames
        it sends text markers so the client can pace the audio to actually-rendered
        video instead of a wall-clock guess --
          video_start                 : real lip-synced video is about to play
          video_clock {frames: N}     : N real frames drained so far this turn
          video_end                   : turn ended; flush any trailing audio
        `real_sent` counts ONLY frames truly drained from the queue (not the
        neutral/held-last frames), so a render stall stops the clock and the
        client's audio waits with it.
        """
        nonlocal sent, q_peak, real_cum
        interval = 1.0 / max(1, fps)
        neutral = engine.neutral_frame()
        # Idle masking: ping-pong through the captured idle loop (blinks/micro-motion) so
        # the priming/between-turns face is alive, not frozen. Falls back to the static
        # neutral portrait if no idle frames were captured (DITTO_IDLE_CAPTURE_CHUNKS<0).
        idle_i = 0
        idle_dir = 1

        def _idle_next() -> bytes:
            nonlocal idle_i, idle_dir
            if not idle_frames:
                return neutral
            if len(idle_frames) == 1:
                return idle_frames[0]
            f = idle_frames[idle_i]
            idle_i += idle_dir
            if idle_i >= len(idle_frames):       # bounce off the end (seamless loop, no wrap jump)
                idle_i, idle_dir = len(idle_frames) - 2, -1
            elif idle_i < 0:
                idle_i, idle_dir = 1, 1
            return f

        last = neutral
        playing = False
        empty_since = None
        real_sent = 0          # real frames emitted this turn (the sync clock)
        last_clock = 0         # last real_sent reported via video_clock
        # Pre-render ("compute-first"): wait until the sentence's frames are actually
        # rendered (a COUNT, not a time gap -- a time gap false-triggers during the
        # renderer's bursty pauses and truncates the sentence). pr_wait_start backs the
        # safety timeout.
        pr_wait_start = None

        async def _mark(payload: dict):
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:  # noqa: BLE001 -- markers are best-effort
                pass

        nxt = loop.time()
        try:
            while not closed.is_set():
                got_real = False
                q_peak = max(q_peak, frame_q.qsize())  # jitter-buffer high-water
                if not playing:
                    out = _idle_next()  # living idle face while priming, not a frozen frame
                    qn = frame_q.qsize()
                    if PRERENDER:
                        # Compute-first: don't play until ~all of this sentence's frames
                        # are queued. expected = audio fed this sentence / 640 (samples per
                        # native frame) / frame_stride (we render only every stride-th).
                        # Waiting on the COUNT (not a time gap) can't false-trigger during
                        # the renderer's burst pauses, so the whole sentence plays.
                        expected = seg_samples / 640.0 / max(1, sdk.frame_stride)
                        if speaking.is_set():
                            pr_wait_start = None       # sentence still arriving -> keep buffering
                            start = False
                        else:
                            now = loop.time()
                            if pr_wait_start is None:   # speech ended -> begin the render wait
                                pr_wait_start = now
                            enough = qn > 0 and qn >= expected * 0.92
                            start = (enough or qn >= PRERENDER_MAX_FRAMES
                                     or (qn > 0 and (now - pr_wait_start) >= PRERENDER_MAX_WAIT_S))
                    else:
                        # Live streaming: start as soon as LEAD_FRAMES buffer OR a short
                        # sentence finished (legacy low-latency path; drifts on a slow GPU).
                        start = (qn >= LEAD_FRAMES) or ((not speaking.is_set()) and qn > 0)
                    if start:
                        playing = True
                        empty_since = None
                        real_sent = 0
                        last_clock = 0
                        pr_wait_start = None
                        # Marker precedes the first real frame -> resets client clock.
                        await _mark({"type": "video_start"})
                if playing:
                    # Play every frame in order; NEVER discard. The old tail-trim capped each
                    # turn at an estimated voice-frame count and cleared the queue -- but on a
                    # sub-realtime GPU the NEXT sentence is already rendering INTO this queue
                    # before the current one finishes, so the clear truncated it (the "avatar
                    # doesn't speak the whole sentence" bug). Since the silent tail is
                    # acceptable, we just drain frames continuously: the voice is frame-clocked
                    # to the real frames either way, so it stays synced -- just delayed, never
                    # cut. video_end fires on a sustained-empty queue (end of the reply).
                    if True:
                        try:
                            last = _frame_to_bytes(frame_q.get_nowait())
                            empty_since = None
                            out = last
                            real_sent += 1
                            real_cum += 1
                            got_real = True
                        except queue.Empty:
                            now = loop.time()
                            if empty_since is None:
                                empty_since = now
                            if now - empty_since >= IDLE_GRACE:
                                # Sustained empty -> turn really ended; re-prime next time.
                                # Hand back to the idle loop so the face keeps living
                                # between turns instead of snapping to a frozen portrait.
                                playing = False
                                last = neutral
                                out = _idle_next()
                                await _mark({"type": "video_end"})
                            else:
                                out = last  # hold last frame over short gaps
                await ws.send_bytes(out)
                sent += 1
                # Report the real-frame clock to the client. Every 2 frames (not 5): the
                # client releases voice paced to this clock, so a coarse interval lets the
                # voice lag the video in steps (5 frames @12.5fps = 0.4s of sawtooth jitter
                # on top of the constant offset). Every 2 (~0.16s) keeps release smooth.
                # Markers are tiny json on localhost -- the higher rate is free.
                if got_real and real_sent - last_clock >= 2:
                    last_clock = real_sent
                    await _mark({"type": "video_clock", "frames": real_sent})
                nxt += interval
                await asyncio.sleep(max(0.0, nxt - loop.time()))
        except Exception:  # noqa: BLE001
            pass

    async def watchdog():
        """Surface silent worker-thread crashes (the SDK hides them until close)
        and log throughput + GPU/render headroom, so delay/lag has a trace.

        render_max vs the per-diffusion budget (valid_clip_len/fps) is the key
        number: if a diffusion takes nearly its whole budget there's no headroom
        and playback stutters. q_peak shows whether the jitter buffer ever fills."""
        nonlocal render_n, render_ms_total, render_ms_max, q_peak
        last_sent = 0
        last_real = 0
        # THE key metric: real_fps = actual rendered frames drained/sec. The pump
        # pads to a steady `fps` with neutral/held frames, so the plain `sent` rate
        # always reads ~fps even when the renderer is behind. If real_fps drops below
        # fps WHILE speaking, the video is falling behind the voice -- that's the lag.
        # feat_* below is only the hubert feature-extraction time (diffusion is in the
        # worker thread, untimed); gpu% is the diffusion-load signal.
        feat_budget_ms = round(STEP / AUDIO_SR * 1000)
        while not closed.is_set():
            await asyncio.sleep(2)
            if sdk.stop_event.is_set():
                logger.error(f"Ditto workers STOPPED. exception={sdk.worker_exception!r}")
            gpu = await asyncio.to_thread(_gpu_stats)
            f_avg = round(render_ms_total / render_n) if render_n else 0
            f_max = round(render_ms_max)
            real_fps = round((real_cum - last_real) / 2.0, 1)
            stages = _profile_drain()  # "stitch=.. warp=.. decode=.. putback=.." avg/max ms
            logger.info(
                f"[ditto] sent={sent} (+{sent - last_sent}/2s) real_fps={real_fps} "
                f"frame_q={frame_q.qsize()} q_peak={q_peak} speaking={speaking.is_set()} "
                f"gpu={gpu['gpu_util']}% vram={gpu['vram_used']}/{gpu['vram_total']} "
                f"temp={gpu['gpu_temp']} feat_avg={f_avg}ms feat_max={f_max}ms "
                f"(budget {feat_budget_ms}ms){(' | ' + stages) if stages else ''} "
                f"workers_stopped={sdk.stop_event.is_set()}"
            )
            # Feed the debug dashboard. real_fps is the keeping-up signal; gpu% the load.
            _STATUS.update({
                "fps": round((sent - last_sent) / 2.0, 1),
                "real_fps": real_fps,
                "sent": sent,
                "workers_stopped": sdk.stop_event.is_set(),
                "q_peak": q_peak,
                "feat_avg_ms": f_avg,
                "feat_max_ms": f_max,
                "feat_budget_ms": feat_budget_ms,
                "gpu_util": gpu["gpu_util"],
                "vram_used": gpu["vram_used"],
                "vram_total": gpu["vram_total"],
                "gpu_temp": gpu["gpu_temp"],
            })
            last_sent = sent
            last_real = real_cum
            render_n = 0
            render_ms_total = 0.0
            render_ms_max = 0.0
            q_peak = 0

    async def feed(pcm: np.ndarray):
        nonlocal audio_buf, started
        if not started:
            audio_buf = np.concatenate([np.zeros((PREPAD,), np.float32), pcm])
            started = True
        else:
            audio_buf = np.concatenate([audio_buf, pcm])
        # Emit every full window, advancing by STEP (online hubert windowing).
        while len(audio_buf) >= SPLIT_LEN:
            await _run_chunk(audio_buf[:SPLIT_LEN])
            audio_buf = audio_buf[STEP:]

    async def flush_turn():
        """At end-of-sentence, feed one valid_clip_len window of silence so the worker
        diffuses the trailing audio that hasn't crossed a boundary yet. A full window is
        what GUARANTEES the whole sentence renders (no truncation): with overlap 10 the
        worker holds up to ~2.8s of un-diffused REAL speech, so this silence is mostly
        pushing that real tail out (it lip-syncs it), not waste. The silence-only frames
        at the very end just let the mouth settle to neutral."""
        n = FLUSH_CHUNKS if FLUSH_CHUNKS > 0 else (valid_clip_len // 5 + 1)
        for _ in range(n):
            await _run_chunk(_silence)

    pump_task = asyncio.create_task(pump())
    watchdog_task = asyncio.create_task(watchdog())
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                evt = json.loads(msg["text"])
                kind = evt.get("type")
                if kind == "config":
                    fps = float(evt.get("fps", fps))
                    # Frame-drop to match the requested OUTPUT fps: render only every
                    # (25 / fps)-th motion frame so a slow GPU keeps up (fps 12.5 -> drop
                    # every other frame). The audio<->frame mapping stays consistent
                    # because the client paces the voice to this same fps.
                    sdk.frame_stride = max(1, round(NATIVE_FPS / fps)) if fps > 0 else 1
                    _STATUS["fps_target"] = fps
                    logger.info(f"[stream] config: fps={fps} -> frame_stride={sdk.frame_stride}")
                elif kind == "speech_start":
                    speaking.set()
                    seg_samples = 0   # new sentence -> reset the expected-frames counter
                elif kind == "speech_end":
                    speaking.clear()
                    # Render this turn's tail now so short replies animate and the
                    # client gets video_clock/video_end markers (no sync fallback).
                    await flush_turn()
                elif kind == "reset":
                    # Barge-in: drop the interrupted turn's buffered audio + any
                    # already-rendered frames so they don't play after the cut,
                    # and force the pump to re-prime cleanly on the next turn.
                    speaking.clear()
                    audio_buf = np.zeros(0, dtype=np.float32)
                    started = False
                    feat_fed = 0  # the worker's buffer is dropped on reset; realign
                    try:
                        while True:
                            frame_q.get_nowait()
                    except queue.Empty:
                        pass
                continue

            data = msg.get("bytes")
            if not data:
                continue
            speaking.set()
            pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            seg_samples += len(pcm)   # track this sentence's audio for the prerender wait
            await feed(pcm)

    except WebSocketDisconnect:
        logger.info("Ditto client disconnected.")
    except Exception:  # noqa: BLE001
        logger.exception("Ditto stream error")
    finally:
        closed.set()
        pump_task.cancel()
        watchdog_task.cancel()
        try:
            await asyncio.to_thread(sdk.close)
        except Exception:  # noqa: BLE001
            pass
        _session_refs = None  # 'global' already declared at session start
        _STATUS.update({"session_active": False, "fps": 0.0, "speaking": False})
        logger.info(f"Session closed (sent {sent} frames).")
        _session_lock.release()


@app.get("/health")
def health():
    return {"ok": engine.sdk is not None, "avatar": str(AVATAR_REF), "size": IMAGE_SIZE}


@app.get("/status")
def status():
    """Live metrics for the pipeline's debug dashboard (read-only)."""
    out = dict(_STATUS)
    # Override queue/speaking/workers with live values when a session is active,
    # so the panel is real-time instead of waiting on the 2s watchdog tick.
    refs = _session_refs
    if refs is not None:
        try:
            out["queue"] = refs["frame_q"].qsize()
            out["speaking"] = refs["speaking"].is_set()
            out["workers_stopped"] = refs["sdk"].stop_event.is_set()
        except Exception:  # noqa: BLE001
            pass
    out["ok"] = engine.sdk is not None
    out.update(_cuda_flags())
    return out


if __name__ == "__main__":
    # Durable per-process log at logs/ditto.log (rotated, full tracebacks, plus
    # uvicorn/asyncio/onnxruntime via the stdlib intercept). log_config=None lets
    # uvicorn's records propagate into loguru instead of its own handlers.
    from log_setup import setup_logging

    setup_logging("ditto")
    uvicorn.run(
        app, host="0.0.0.0", port=int(os.getenv("DITTO_PORT", "8002")),
        ws_ping_interval=None, ws_ping_timeout=None, log_config=None,
        # The dashboard polls /status every 0.5s; without this, uvicorn's access
        # log floods ditto.log and buries the real events (model load, watchdog,
        # worker crashes) that we actually need to see. Those come via loguru.
        access_log=False,
    )
