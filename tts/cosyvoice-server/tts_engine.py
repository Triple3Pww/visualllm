"""
Local TTS engine wrapper around CosyVoice2 (FunAudioLLM).

Loads the model once, registers a fixed reference voice, and exposes a simple
synthesize(text) -> (waveform, sample_rate) API. Used by test_en.py, test_zh.py,
benchmark.py, and app.py so they all share one engine implementation.

Voice: the reference voice is CosyVoice's bundled `asset/zero_shot_prompt.wav`
(a female Mandarin speaker) plus its transcript. Swap PROMPT_WAV / PROMPT_TEXT
to change the forecaster's voice — no other code changes needed.

Language routing: Chinese (and any CJK) text uses inference_zero_shot; Latin /
English text uses inference_cross_lingual (the upstream-recommended path for a
target language that differs from the prompt's language).
"""
from __future__ import annotations

import os
import sys
import re
import logging
from pathlib import Path

# Quiet the HuggingFace tokenizers fork warning (CosyVoice forks after tokenizing).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- locate the cloned CosyVoice repo and put it (and Matcha-TTS) on sys.path -
HERE = Path(__file__).resolve().parent
COSY_DIR = HERE / "CosyVoice"
MATCHA_DIR = COSY_DIR / "third_party" / "Matcha-TTS"
for p in (str(COSY_DIR), str(MATCHA_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

DEFAULT_MODEL_DIR = str(COSY_DIR / "pretrained_models" / "CosyVoice2-0.5B")

# --- fixed reference voice ---------------------------------------------------
# Overridable via env so the registered voice can be swapped without editing source:
#   COSYVOICE_PROMPT_WAV  = path to the reference clip (clean, mono)
#   COSYVOICE_PROMPT_TEXT = the EXACT transcript of that clip (zero-shot needs it)
# BASELINE (2026-07-02): the "pro" female AI-assistant voice (asset/pro_ref.wav). zero_shot
# clones the reference's RHYTHM, and this clip is naturally fluid, so zh comes out smooth
# (~1.3 pauses/sentence vs the old "weather" clip's ~3.8) -- fewer than English -- with no
# pause-trimming needed. The old "希望你以后..." weather speaker is asset/zero_shot_prompt.wav.
PROMPT_WAV = os.environ.get(
    "COSYVOICE_PROMPT_WAV", str(COSY_DIR / "asset" / "pro_ref.wav")
)
PROMPT_TEXT = os.environ.get(
    "COSYVOICE_PROMPT_TEXT", "你好，我是你的AI虚拟助手，很高兴见到你。今天天气不错，有什么我可以帮你的"
)
SPK_ID = os.environ.get("COSYVOICE_SPK_ID", "weather")

# CosyVoice3 only: the instruct prefix that must precede <|endofprompt|>. Upstream's own
# cosyvoice3_example uses exactly this string (example.py:76,81). The marker SEPARATES the
# prefix from the reference transcript (llm.py:591) -- putting it at the END of prompt_text
# leaves an empty transcript.
V3_INSTRUCT = os.environ.get("COSYVOICE_V3_INSTRUCT", "You are a helpful assistant.<|endofprompt|>")

_CJK = re.compile(r"[㐀-鿿豈-﫿぀-ヿ]")


def is_cjk(text: str) -> bool:
    """True if the text contains any Chinese/Japanese characters."""
    return bool(_CJK.search(text))


class TTSEngine:
    def __init__(self, model_dir: str | None = None, fp16: bool = False, load_vllm: bool = False,
                 load_trt: bool = False):
        import torch
        from cosyvoice.cli.cosyvoice import AutoModel

        self.model_dir = model_dir or os.environ.get("COSYVOICE_MODEL_DIR", DEFAULT_MODEL_DIR)
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(
                f"Model not found at {self.model_dir}. Download it with:\n"
                f"  python -c \"from modelscope import snapshot_download; "
                f"snapshot_download('iic/CosyVoice2-0.5B', local_dir='{self.model_dir}')\""
            )

        # CosyVoice2 uses CUDA when available, else CPU. MPS is not used by the
        # upstream model, so on Apple Silicon this runs on CPU.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("Loading CosyVoice from %s (device=%s)", self.model_dir, self.device)

        # load_vllm: swap the autoregressive LLM onto vLLM (the real fix for first-chunk latency
        # -- the LLM token-gen is the ~3s bottleneck). Off by default (COSYVOICE_VLLM=1 to enable);
        # the Windows server stays on the PyTorch path. Requires the vLLM env (Linux/WSL).
        # AutoModel dispatches on the yaml in model_dir, so COSYVOICE_MODEL_DIR alone selects
        # CosyVoice2 vs CosyVoice3. No load_jit: CosyVoice3.__init__ has no such parameter, and
        # this passed False for it anyway.
        self.model = AutoModel(model_dir=self.model_dir, load_trt=load_trt,
                               load_vllm=load_vllm, fp16=fp16)
        self.sample_rate = self.model.sample_rate

        # CosyVoice3's LM asserts <|endofprompt|> is among the tokens it sees (llm.py:479).
        # zh rides the zero_shot path, which keeps prompt_text (our COSYVOICE_PROMPT_TEXT carries
        # the marker). en rides cross_lingual, which DELETES prompt_text -- so on v3 the marker
        # must ride on the text itself, the way upstream's instruct strings do. v2 must never see
        # it (unknown token, and its LM has no such requirement).
        self._is_v3 = self.model.__class__.__name__ == "CosyVoice3"

        # Register the reference voice once; subsequent calls reuse it by id.
        ok = self.model.add_zero_shot_spk(PROMPT_TEXT, PROMPT_WAV, SPK_ID)
        if ok is not True:
            logging.warning("add_zero_shot_spk returned %r; falling back to per-call prompt", ok)
        self._spk_ready = ok is True

        # Traditional -> Simplified before synthesis. CosyVoice's text frontend garbles long
        # TRADITIONAL zh input (spoken output degrades into noise past ~10 chars; Simplified is
        # clean -- reproduced locally 2026-07-11: the same sentence in Traditional transcribes
        # back to garbage while its t2s twin is near-perfect, and the Traditional audio runs
        # ~1-2s longer). Simplified and Traditional are the SAME spoken Mandarin, so the
        # conversion is INAUDIBLE. This pipeline feeds Traditional (llama-4-scout output, sherpa
        # STT -> Traditional), so it is on by default. COSYVOICE_T2S=0 disables; missing opencc
        # -> no-op (old behavior), never a crash. Credit: chiashengchen/CosyVoice root-cause find.
        self._t2s_cc = None
        if os.environ.get("COSYVOICE_T2S", "1").lower() in ("1", "true", "yes", "on"):
            try:
                from opencc import OpenCC
                self._t2s_cc = OpenCC("t2s")
            except Exception as e:  # noqa: BLE001 -- missing opencc = keep old behavior
                logging.warning("OpenCC unavailable, Traditional zh will NOT be converted: %r", e)

    def _to_simplified(self, text: str) -> str:
        """Traditional -> Simplified (inaudible) to dodge the frontend's long-Traditional garble.

        A no-op on Latin/English text (opencc leaves non-Chinese untouched) and when the
        converter is disabled/unavailable, so the en cross-lingual path is unaffected.
        """
        return self._t2s_cc.convert(text) if self._t2s_cc else text

    def _xlingual_text(self, text: str) -> str:
        """cross_lingual drops prompt_text, so v3's instruct prefix rides on the text instead."""
        return f"{V3_INSTRUCT}{text}" if self._is_v3 else text

    def _apply_first_hop(self, cjk: bool):
        """Per-language first-hop, set per request just before inference.

        Capping the FIRST streaming chunk after fewer speech tokens cuts Chinese's slow
        opening (zh commits to a BIGGER opening chunk: TTFB ~2.3s vs en ~1.1s). But it HURTS
        English: the extra turn-start vocoder work starves MuseTalk's first-frame render on the
        shared GPU, so the avatar lips start ~1s LATER (measured 0.7s->1.7s lip-start lag) -- for
        ~no en benefit (en's opening is already small; its TTFO lever is the pipeline's clause
        split, not the hop). So the hop applies to zh ONLY; en stays 0 to protect lip-sync.
          COSYVOICE_FIRST_HOP_ZH  (default = legacy COSYVOICE_FIRST_HOP, else 0)
          COSYVOICE_FIRST_HOP_EN  (default 0 = off)
        """
        legacy = os.getenv("COSYVOICE_FIRST_HOP", "0") or "0"
        raw = os.getenv("COSYVOICE_FIRST_HOP_ZH", legacy) if cjk else os.getenv("COSYVOICE_FIRST_HOP_EN", "0")
        try:
            self.model.model._first_hop = int(raw or "0")
        except Exception:  # noqa: BLE001 -- best-effort; leave the model's own default
            pass

    def synthesize(self, text: str, speed: float = 1.0):
        """Return (waveform_tensor[1, N], sample_rate). Non-streaming."""
        import torch

        text = (text or "").strip()
        if not text:
            raise ValueError("text is empty")

        text = self._to_simplified(text)
        self._apply_first_hop(is_cjk(text))
        if is_cjk(text):
            if self._spk_ready:
                gen = self.model.inference_zero_shot(
                    text, "", "", zero_shot_spk_id=SPK_ID, stream=False, speed=speed
                )
            else:
                gen = self.model.inference_zero_shot(
                    text, PROMPT_TEXT, PROMPT_WAV, stream=False, speed=speed
                )
        else:
            # Cross-lingual: English target voiced with the Mandarin reference.
            spk = SPK_ID if self._spk_ready else ""
            gen = self.model.inference_cross_lingual(
                self._xlingual_text(text), PROMPT_WAV, zero_shot_spk_id=spk, stream=False, speed=speed
            )

        chunks = [out["tts_speech"] for out in gen]
        if not chunks:
            raise RuntimeError("CosyVoice produced no audio")
        return torch.concat(chunks, dim=1), self.sample_rate

    def _trim_lead_in(self, chunks):
        """Drop the inaudible lead-in the zero-shot synth prepends before the first word.

        Every zh piece opens with a low-level breath (~0.2-0.6s, far below speech level). It is
        inaudible, but it IS audio: the listener waits through it before hearing a word, and the
        avatar -- which lip-syncs off a Whisper of the waveform -- moves the mouth over it. It is
        pure TTFO dead time (measured median 0.23s, up to 0.60s).

        Trimming it HERE, on whole tensors, is the only safe place: the same trim attempted on the
        pipecat client's byte stream crashed on aiohttp's odd-sized chunks (PROBLEMS-AND-FIXES P34).
        Bounded by _MAX_S so a mis-set threshold can never swallow real speech, and it keeps a short
        pre-roll so the first phoneme's attack is not clipped. COSYVOICE_TRIM_LEAD=0 disables.
        """
        import numpy as np
        thr = float(os.getenv("COSYVOICE_TRIM_LEAD_THRESH", "0.02"))
        max_s = float(os.getenv("COSYVOICE_TRIM_LEAD_MAX_S", "1.5"))
        pre_s = float(os.getenv("COSYVOICE_TRIM_LEAD_PREROLL_S", "0.03"))
        dropped = 0                                   # samples discarded so far (this utterance)
        started = False
        for wav, sr_ in chunks:
            if started:
                yield wav, sr_
                continue
            # Frame RMS, not per-sample abs: a breath carries occasional spikes above any
            # sample threshold while staying inaudible, so a per-sample test leaves it in
            # (measured: 3 of 8 pieces kept 0.35-0.46s of it). RMS over 10ms is the same
            # measure the ear -- and the playout beacon -- actually responds to.
            x = wav.reshape(-1).detach().cpu().numpy().astype(np.float32)
            fr = max(1, int(sr_ * 0.01))
            nf = len(x) // fr
            voiced = -1
            for i in range(nf):
                f = x[i * fr:(i + 1) * fr]
                if float(np.sqrt(np.mean(f * f))) > thr:
                    voiced = i * fr
                    break
            if voiced < 0:                            # whole chunk inaudible
                dropped += len(x)
                if dropped / sr_ >= max_s:            # cap reached -> stop trimming, pass it through
                    started = True
                    yield wav, sr_
                continue                              # ... otherwise drop it entirely
            first = max(0, voiced - int(pre_s * sr_))
            if (dropped + first) / sr_ > max_s:
                first = 0                             # cap reached mid-chunk -> keep it intact
            started = True
            yield wav[..., first:], sr_

    def _squeeze_silence(self, chunks):
        """Streaming pause-compressor: cap over-long internal silences. OFF by default now.

        NOTE (baseline 2026-07-02): this is a leftover band-aid from when the zh reference
        was the gappy "weather" clip (~57% voiced / ~3.8 pauses/sentence). The "pro" baseline
        voice is naturally fluid (~1.3 pauses, fewer than English), so trimming is unnecessary
        and OFF by default. Re-enable for a gappy voice with COSYVOICE_SILENCE_CAP_S=<seconds>
        (e.g. 0.15): it caps any silent run to that length, keeping short pauses but removing
        the excess -- only ever drops near-silent frames, never speech. Stateful across the
        streamed chunks so a gap spanning a chunk boundary is still capped.
        """
        import torch
        import numpy as np
        sr = self.sample_rate
        frame = int(sr * 0.02)                                   # 20 ms granularity
        cap_frames = max(1, round(float(os.getenv("COSYVOICE_SILENCE_CAP_S", "0")) / 0.02))
        thr = float(os.getenv("COSYVOICE_SILENCE_THR", "0.015"))  # abs RMS silence floor
        carry = np.zeros(0, dtype=np.float32)
        sil = 0
        for wav, sr_ in chunks:
            a = np.concatenate([carry, wav.reshape(-1).detach().cpu().numpy().astype(np.float32)])
            nf = len(a) // frame
            carry = a[nf * frame:]
            keep = []
            for i in range(nf):
                f = a[i * frame:(i + 1) * frame]
                if float(np.sqrt(np.mean(f * f))) < thr:
                    sil += 1
                    if sil <= cap_frames:
                        keep.append(f)                            # keep up to the cap, drop the rest
                else:
                    sil = 0
                    keep.append(f)
            if keep:
                yield torch.from_numpy(np.concatenate(keep)).unsqueeze(0), sr_
        if carry.size:
            yield torch.from_numpy(carry).unsqueeze(0), self.sample_rate

    def synthesize_stream(self, text: str, speed: float = 1.0):
        """Yield (waveform_tensor[1, N], sample_rate) chunks as they synthesize.

        Same voice/language routing as synthesize(), but stream=True so the first
        chunk is emitted before the whole utterance is done -- the path the realtime
        pipeline (Pipecat -> avatar) needs to start lip-syncing within the TTFO budget.

        The zh pause-trimmer (_squeeze_silence) is OFF by default in this baseline (the pro
        voice doesn't need it); enable it for a gappier voice with COSYVOICE_SILENCE_CAP_S>0.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("text is empty")

        text = self._to_simplified(text)
        cjk = is_cjk(text)
        self._apply_first_hop(cjk)
        if cjk:
            if self._spk_ready:
                gen = self.model.inference_zero_shot(
                    text, "", "", zero_shot_spk_id=SPK_ID, stream=True, speed=speed
                )
            else:
                gen = self.model.inference_zero_shot(
                    text, PROMPT_TEXT, PROMPT_WAV, stream=True, speed=speed
                )
        else:
            spk = SPK_ID if self._spk_ready else ""
            gen = self.model.inference_cross_lingual(
                self._xlingual_text(text), PROMPT_WAV, zero_shot_spk_id=spk, stream=True, speed=speed
            )

        raw = ((out["tts_speech"], self.sample_rate) for out in gen)
        squeeze = cjk and float(os.getenv("COSYVOICE_SILENCE_CAP_S", "0")) > 0
        out_stream = self._squeeze_silence(raw) if squeeze else raw
        # Strip the leading breath before the first word (TTFO dead time; see _trim_lead_in).
        if (os.getenv("COSYVOICE_TRIM_LEAD", "1") or "1") not in ("0", "false", "False"):
            out_stream = self._trim_lead_in(out_stream)
        yield from out_stream


# Module-level singleton so importing scripts share one loaded model.
_ENGINE: TTSEngine | None = None


def get_engine() -> TTSEngine:
    global _ENGINE
    if _ENGINE is None:
        # Lever 1 (P27 follow-on): run the flow-matching estimator (20 passes/chunk, the remaining
        # first-chunk compute after CUDA graphs) on TensorRT instead of PyTorch. COSYVOICE_FLOW_TRT=1
        # -> load_trt=True; CosyVoice auto-builds flow.decoder.estimator.<prec>.mygpu.plan from the
        # shipped fp32 onnx on first load. fp16 is DECOUPLED (COSYVOICE_FP16): on TRT 11 the fp16
        # builder flag is gone (strongly-typed only), so fp16 needs an fp16 onnx -- deferred. fp32 TRT
        # builds today (kernel-fusion win, zero accuracy risk). fp16 does NOT touch vLLM (llm_job
        # disables its autocast when vllm is loaded). Bare fp16 was once measured useless, but that
        # was pre-vLLM when AR token-gen dominated; the flow is now the FLOP-bound remainder.
        import torch
        flow_trt = (os.environ.get("COSYVOICE_FLOW_TRT", "0").lower() in ("1", "true", "yes", "on")
                    and torch.cuda.is_available())
        fp16 = (os.environ.get("COSYVOICE_FP16", "0").lower() in ("1", "true", "yes", "on")
                and torch.cuda.is_available())
        load_vllm = (os.environ.get("COSYVOICE_VLLM", "0").lower() in ("1", "true", "yes", "on")
                     and torch.cuda.is_available())
        _ENGINE = TTSEngine(fp16=fp16, load_vllm=load_vllm, load_trt=flow_trt)
    return _ENGINE
