"""MuseTalk real-time lip-sync server (FastAPI + websocket).

Protocol (matches local_services/musetalk_video.py):
  client -> server:
    text json {"type":"config","fps":25}
    text json {"type":"speech_start"} / {"type":"speech_end"} / {"type":"reset"}
    binary: 16-bit PCM mono @16 kHz audio chunks (TTS audio, resampled by client)
  server -> client:
    binary: raw RGB frame buffers (IMAGE_SIZE*IMAGE_SIZE*3 bytes) at `fps`

Implementation notes
--------------------
This drives MuseTalk v1.5 locally on the GPU. It reuses the upstream model code
in ``vendor/MuseTalk`` (cloned next to this file) but replaces two things so it
runs on this machine:

* **Streaming** instead of whole-file inference. Audio arrives in PCM chunks; we
  buffer it and run UNet/VAE on fixed segments (``SEG_FRAMES`` frames each), so
  the avatar starts talking mid-utterance. Idle neutral frames keep the WebRTC
  video track alive between turns.
* **No mmpose/DWPose** for avatar preparation. Upstream uses DWPose (needs
  mmcv/mmpose, which require a CUDA compiler that isn't installed here). DWPose
  is only used to get the 68 iBUG face landmarks (keypoints[23:91]); we get the
  same 68 landmarks from ``face_alignment`` (pure-torch, pip-only) and feed them
  through MuseTalk's exact bbox math. Preparation is one-time and cached.

The realtime loop uses only the VAE decoder + UNet + Whisper — no landmark model
— so ``face_alignment`` is only imported during preparation.

Setup is handled by the project: a dedicated ``musetalk`` conda env (cu128 torch
+ diffusers/transformers + face_alignment) and weights under
``vendor/MuseTalk/models``. Run with:
    conda run -n musetalk python -m local_services.musetalk_server.app
VRAM: ~4-6 GB.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import asyncio
from pathlib import Path

# face_alignment 1.5 wraps its net in torch.compile, which needs Triton (absent
# on Windows). Disable TorchDynamo so it runs eagerly — MuseTalk isn't compiled.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger

# --- paths -----------------------------------------------------------------
SERVER_DIR = Path(__file__).resolve().parent
MUSETALK_ROOT = SERVER_DIR / "vendor" / "MuseTalk"
MODELS_DIR = MUSETALK_ROOT / "models"
CACHE_DIR = SERVER_DIR / "avatar_cache"

# AVATAR_REF is given relative to the project root; resolve before we chdir.
AVATAR_REF = Path(os.getenv("AVATAR_REF", "assets/avatar.png")).resolve()

# --- knobs (env-overridable) ----------------------------------------------
IMAGE_SIZE = int(os.getenv("MUSETALK_SIZE", "512"))   # output frame is SIZE x SIZE
AUDIO_SR = 16000                                       # Whisper expects 16 kHz
DEFAULT_FPS = int(os.getenv("MUSETALK_FPS", "20"))  # ~32ms/frame GPU floor -> 20fps keeps realtime headroom
SEG_FRAMES = int(os.getenv("MUSETALK_SEG_FRAMES", "8"))   # frames per UNet segment
IDLE_FPS = int(os.getenv("MUSETALK_IDLE_FPS", "10"))       # neutral-frame rate between turns
BATCH_SIZE = int(os.getenv("MUSETALK_BATCH", "8"))
PAD_LEFT = int(os.getenv("MUSETALK_PAD_LEFT", "2"))
PAD_RIGHT = int(os.getenv("MUSETALK_PAD_RIGHT", "2"))
EXTRA_MARGIN = int(os.getenv("MUSETALK_EXTRA_MARGIN", "10"))
PARSING_MODE = os.getenv("MUSETALK_PARSING_MODE", "jaw")
# Cap the base-portrait resolution. Output is IMAGE_SIZE^2, so a huge source only
# slows the per-frame full-frame compositing (PIL). Keeps the face well above the
# 256px VAE crop while making blending realtime.
BASE_MAX = int(os.getenv("MUSETALK_BASE_MAX", "768"))

app = FastAPI(title="MuseTalk realtime")


class MuseTalkEngine:
    """MuseTalk models + a prepared (cached) avatar, with streaming inference."""

    def __init__(self, ref_path: Path, size: int, fps: int):
        self.ref_path = ref_path
        self.size = size
        self.fps = fps
        self._ready = False
        self.idx = 0  # base-frame cursor (cycles for video refs; static for a photo)

    # --- one-time load ----------------------------------------------------
    def load(self):
        if self._ready:
            return
        if not MODELS_DIR.exists():
            raise RuntimeError(
                f"MuseTalk weights not found at {MODELS_DIR}. Run download_weights "
                f"(see local_services/musetalk_server/vendor/MuseTalk)."
            )

        import torch
        import cv2  # noqa: F401  (ensures cv2 import errors surface early)
        from transformers import WhisperModel

        # MuseTalk's checkpoints are legacy/pickled; PyTorch 2.6+ defaults
        # torch.load to weights_only=True and rejects them. Restore the old
        # behavior for the vendored loaders (trusted local weights).
        if not getattr(torch.load, "_musetalk_patched", False):
            _orig_load = torch.load

            def _load(*a, **k):
                k.setdefault("weights_only", False)
                return _orig_load(*a, **k)

            _load._musetalk_patched = True
            torch.load = _load

        # MuseTalk uses CWD-relative model paths; run from its root.
        os.chdir(MUSETALK_ROOT)
        if str(MUSETALK_ROOT) not in sys.path:
            sys.path.insert(0, str(MUSETALK_ROOT))

        from musetalk.utils.utils import load_all_model
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.face_parsing import FaceParsing

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Fixed input shapes -> let cudnn autotune; allow TF32 for the fp32 bits.
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info(f"Loading MuseTalk v1.5 models on {self.device} …")

        vae, unet, pe = load_all_model(
            unet_model_path=str(MODELS_DIR / "musetalkV15" / "unet.pth"),
            vae_type="sd-vae",
            unet_config=str(MODELS_DIR / "musetalkV15" / "musetalk.json"),
            device=self.device,
        )
        self.timesteps = torch.tensor([0], device=self.device)
        self.pe = pe.half().to(self.device)
        vae.vae = vae.vae.half().to(self.device)
        unet.model = unet.model.half().to(self.device)
        self.vae, self.unet = vae, unet
        self.weight_dtype = unet.model.dtype

        self.audio_processor = AudioProcessor(feature_extractor_path=str(MODELS_DIR / "whisper"))
        self.feature_extractor = self.audio_processor.feature_extractor
        whisper = WhisperModel.from_pretrained(str(MODELS_DIR / "whisper"))
        self.whisper = whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

        self.fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

        self._prepare_avatar()
        self._neutral = self._frame_to_bytes(self.frame_cycle[0])
        self._ready = True
        logger.info(f"MuseTalk ready. {len(self.frame_cycle)} base frame(s) prepared.")

    # --- avatar preparation (mmpose-free, cached) -------------------------
    def _avatar_key(self) -> str:
        st = self.ref_path.stat()
        raw = f"{self.ref_path}|{st.st_size}|{int(st.st_mtime)}|v15|m{EXTRA_MARGIN}|{PARSING_MODE}|b{BASE_MAX}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _prepare_avatar(self):
        import cv2

        if not self.ref_path.exists():
            raise RuntimeError(
                f"Avatar reference not found: {self.ref_path}. Put a front-facing "
                f"portrait at assets/avatar.png (see assets/README.md)."
            )

        cache = CACHE_DIR / self._avatar_key()
        if (cache / "materials.pkl").exists():
            logger.info(f"Loading cached avatar from {cache}")
            with open(cache / "materials.pkl", "rb") as f:
                mats = pickle.load(f)
            self.frame_cycle = mats["frames"]
            self.coord_cycle = mats["coords"]
            self.mask_cycle = mats["masks"]
            self.mask_coords_cycle = mats["mask_coords"]
            self.latent_cycle = self.torch.load(cache / "latents.pt", map_location=self.device)
            return

        logger.info("Preparing avatar (one-time): detecting landmarks + encoding latents …")
        from musetalk.utils.blending import get_image_prepare_material

        # Load reference as a list of BGR frames (image -> 1 frame; video -> N).
        frames = self._read_ref_frames(self.ref_path)
        coords = self._landmark_bboxes(frames)

        valid_frames, valid_coords, latents = [], [], []
        for bbox, frame in zip(coords, frames):
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            y2 = min(y2 + EXTRA_MARGIN, frame.shape[0])      # v15 extra margin
            bbox = (x1, y1, x2, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents.append(self.vae.get_latents_for_unet(resized))
            valid_frames.append(frame)
            valid_coords.append(bbox)

        if not valid_frames:
            raise RuntimeError(
                "No face detected in the avatar reference. Use a clear, front-facing "
                "portrait at assets/avatar.png."
            )

        # Ping-pong the cycle so a short clip loops smoothly (a photo stays static).
        self.frame_cycle = valid_frames + valid_frames[::-1]
        self.coord_cycle = valid_coords + valid_coords[::-1]
        self.latent_cycle = latents + latents[::-1]

        self.mask_cycle, self.mask_coords_cycle = [], []
        for frame, bbox in zip(self.frame_cycle, self.coord_cycle):
            mask, crop_box = get_image_prepare_material(
                frame, list(bbox), fp=self.fp, mode=PARSING_MODE
            )
            self.mask_cycle.append(mask)
            self.mask_coords_cycle.append(crop_box)

        cache.mkdir(parents=True, exist_ok=True)
        with open(cache / "materials.pkl", "wb") as f:
            pickle.dump(
                {
                    "frames": self.frame_cycle,
                    "coords": self.coord_cycle,
                    "masks": self.mask_cycle,
                    "mask_coords": self.mask_coords_cycle,
                },
                f,
            )
        self.torch.save(self.latent_cycle, cache / "latents.pt")
        logger.info(f"Avatar materials cached to {cache}")

    def _read_ref_frames(self, path: Path):
        import cv2

        def _cap(fr):
            h, w = fr.shape[:2]
            m = max(h, w)
            if m > BASE_MAX:
                s = BASE_MAX / m
                fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            return fr

        ext = path.suffix.lower()
        if ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            cap = cv2.VideoCapture(str(path))
            frames = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(_cap(fr))
                if len(frames) >= 200:  # cap clip length to bound prep time/VRAM
                    break
            cap.release()
            if not frames:
                raise RuntimeError(f"Could not read frames from {path}")
            return frames
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"Could not read image {path}")
        return [_cap(img)]

    def _landmark_bboxes(self, frames):
        """68-landmark bbox per frame via face_alignment — MuseTalk's exact math,
        DWPose replaced. Returns a list of (x1,y1,x2,y2) or None."""
        import face_alignment

        lt = getattr(face_alignment.LandmarksType, "TWO_D",
                     getattr(face_alignment.LandmarksType, "_2D", None))
        fa = face_alignment.FaceAlignment(
            lt, flip_input=False,
            device="cuda" if self.device.type == "cuda" else "cpu",
        )

        out = []
        for frame in frames:
            rgb = frame[:, :, ::-1]  # BGR -> RGB for face_alignment
            preds = fa.get_landmarks(rgb)
            if not preds:
                out.append(None)
                continue
            lm = preds[0].astype(np.int32)                       # (68,2) iBUG-68
            half_y = lm[29][1]                                   # nose bridge (idx 29)
            half_dist = int(np.max(lm[:, 1]) - half_y)
            upper = max(0, half_y - half_dist)
            x1, x2 = int(np.min(lm[:, 0])), int(np.max(lm[:, 0]))
            y1, y2 = int(upper), int(np.max(lm[:, 1]))
            if x1 < 0 or x2 - x1 <= 0 or y2 - y1 <= 0:
                out.append(None)
            else:
                out.append((x1, y1, x2, y2))
        del fa  # free the landmark model; the realtime loop never needs it
        return out

    # --- realtime inference ----------------------------------------------
    def samples_per_frame(self, fps: int) -> int:
        return int(AUDIO_SR / fps)

    def reset_idx(self):
        self.idx = 0

    def neutral_frame(self) -> bytes:
        return self._neutral

    def _frame_to_bytes(self, frame_bgr) -> bytes:
        import cv2

        out = cv2.resize(frame_bgr, (self.size, self.size), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()

    def _audio_features(self, audio: np.ndarray):
        seg_len = 30 * AUDIO_SR
        segments = [audio[i:i + seg_len] for i in range(0, len(audio), seg_len)] or [audio]
        feats = []
        for seg in segments:
            af = self.feature_extractor(
                seg, return_tensors="pt", sampling_rate=AUDIO_SR
            ).input_features.to(self.weight_dtype)
            feats.append(af)
        return feats, len(audio)

    def _composite(self, res_bgr: np.ndarray, idx: int) -> bytes:
        import cv2
        from musetalk.utils.blending import get_image_blending

        bbox = self.coord_cycle[idx]
        x1, y1, x2, y2 = bbox
        ori = self.frame_cycle[idx].copy()
        face = cv2.resize(res_bgr.astype(np.uint8), (x2 - x1, y2 - y1))
        combine = get_image_blending(
            ori, face, bbox, self.mask_cycle[idx], self.mask_coords_cycle[idx]
        )
        return self._frame_to_bytes(combine)

    def render_segment(self, audio: np.ndarray) -> list[bytes]:
        """One audio segment (float32 [-1,1] @16k) -> list of RGB frame buffers.

        Runs on a worker thread (GPU-bound). Index cursor and base frames are
        cycled per produced frame so latents and composites stay aligned.
        """
        torch = self.torch
        prof = os.getenv("MUSETALK_PROFILE")
        import time as _t

        t0 = _t.time()
        try:
            feats, length = self._audio_features(audio)
            t_feat = _t.time()
            chunks = self.audio_processor.get_whisper_chunk(
                feats, self.device, self.weight_dtype, self.whisper, length,
                fps=self.fps, audio_padding_length_left=PAD_LEFT,
                audio_padding_length_right=PAD_RIGHT,
            )
            t_whisper = _t.time()
        except (AssertionError, SystemExit, Exception):  # noqa: BLE001
            logger.exception("whisper chunking failed; dropping segment")
            return []

        L = len(self.latent_cycle)
        out: list[bytes] = []
        gpu_s = 0.0
        comp_s = 0.0
        with torch.no_grad():
            for i in range(0, len(chunks), BATCH_SIZE):
                w_batch = chunks[i:i + BATCH_SIZE].to(self.device)
                n = w_batch.shape[0]
                idxs = [(self.idx + k) % L for k in range(n)]
                latent_batch = torch.cat([self.latent_cycle[x] for x in idxs], dim=0).to(
                    device=self.device, dtype=self.unet.model.dtype
                )
                audio_feat = self.pe(w_batch)
                pred = self.unet.model(
                    latent_batch, self.timesteps, encoder_hidden_states=audio_feat
                ).sample
                pred = pred.to(dtype=self.vae.vae.dtype)
                recon = self.vae.decode_latents(pred)  # [n,256,256,3] BGR uint8
                tg = _t.time()
                gpu_s += tg - (t_whisper if i == 0 else tc)
                for k in range(n):
                    out.append(self._composite(recon[k], idxs[k]))
                tc = _t.time()
                comp_s += tc - tg
                self.idx = (self.idx + n) % L
        if prof:
            logger.info(
                f"[profile] feat={1000*(t_feat-t0):.0f}ms "
                f"whisper={1000*(t_whisper-t_feat):.0f}ms "
                f"gpu={1000*gpu_s:.0f}ms composite={1000*comp_s:.0f}ms "
                f"-> {len(out)} frames"
            )
        return out


engine = MuseTalkEngine(AVATAR_REF, IMAGE_SIZE, DEFAULT_FPS)


@app.on_event("startup")
def _startup():
    engine.load()


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    fps = engine.fps
    spf = engine.samples_per_frame(fps)
    seg_samples = spf * SEG_FRAMES
    audio_buf = np.zeros(0, dtype=np.float32)
    # Bounded queue of rendered frames; the pump drains it at a STEADY fps.
    out_q: asyncio.Queue = asyncio.Queue(maxsize=600)
    closed = asyncio.Event()
    loop = asyncio.get_event_loop()

    async def pump():
        """Emit a steady `fps` video stream. WebRTC/mobile decoders freeze on
        bursty input, so we pace output: send the next rendered frame if one is
        ready, otherwise repeat the last frame to hold a smooth track."""
        interval = 1.0 / max(1, fps)
        last = engine.neutral_frame()
        nxt = loop.time()
        try:
            while not closed.is_set():
                try:
                    last = out_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                await ws.send_bytes(last)
                nxt += interval
                await asyncio.sleep(max(0.0, nxt - loop.time()))
        except Exception:  # noqa: BLE001
            pass

    async def render(segment: np.ndarray) -> int:
        frames = await asyncio.to_thread(engine.render_segment, segment)
        for f in frames:
            if out_q.full():  # stay realtime: drop oldest rather than lag behind
                try:
                    out_q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            out_q.put_nowait(f)
        return len(frames)

    pump_task = asyncio.create_task(pump())
    turn_frames = 0

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                evt = json.loads(msg["text"])
                kind = evt.get("type")
                if kind == "config":
                    fps = int(evt.get("fps", fps))
                    engine.fps = fps
                    spf = engine.samples_per_frame(fps)
                    seg_samples = spf * SEG_FRAMES
                    logger.info(f"[stream] config: fps={fps}")
                elif kind == "speech_start":
                    turn_frames = 0
                elif kind in ("reset", "speech_end"):
                    if kind == "speech_end" and len(audio_buf) >= spf:
                        pad = (-len(audio_buf)) % spf
                        seg = (
                            np.concatenate([audio_buf, np.zeros(pad, np.float32)])
                            if pad else audio_buf
                        )
                        turn_frames += await render(seg)
                    audio_buf = np.zeros(0, dtype=np.float32)
                    engine.reset_idx()
                    if kind == "speech_end":
                        logger.info(f"[stream] turn rendered {turn_frames} frames")
                        out_q.put_nowait(engine.neutral_frame())  # close mouth after speaking
                continue

            data = msg.get("bytes")
            if not data:
                continue
            pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buf = np.concatenate([audio_buf, pcm])
            while len(audio_buf) >= seg_samples:
                seg = audio_buf[:seg_samples]
                audio_buf = audio_buf[seg_samples:]
                turn_frames += await render(seg)

    except WebSocketDisconnect:
        logger.info("MuseTalk client disconnected.")
    except Exception:  # noqa: BLE001
        logger.exception("MuseTalk stream error")
    finally:
        closed.set()
        await asyncio.gather(pump_task, return_exceptions=True)


@app.get("/health")
def health():
    return {"ok": engine._ready, "avatar": str(AVATAR_REF), "size": IMAGE_SIZE}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
