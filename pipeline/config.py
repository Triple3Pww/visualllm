"""Central configuration: keys, model/voice ids, and the language switch.

One pure stack — Deepgram STT -> OpenRouter LLM -> CosyVoice TTS -> MuseTalk avatar.
Everything is read from .env so keys stay out of git. Behavioral knobs:
LANGUAGE (en/zh/th), TTFO_TARGET_SECONDS, TTS_PROVIDER, and the MUSETALK_* avatar knobs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name, default)
    return val.strip() if isinstance(val, str) else val


def _get_float(name: str, default: str) -> float:
    """Parse a numeric env var, falling back to `default` on blank/garbage.

    os.getenv returns "" (not the default) when a key is present-but-empty in
    .env, so a stray `FOO=` would make float("") blow up at import. Fall back.
    """
    raw = _get(name)
    if raw:
        try:
            return float(raw)
        except ValueError:
            import warnings

            warnings.warn(
                f"{name}={raw!r} is not a number; using default {default}.",
                stacklevel=2,
            )
    return float(default)


def _get_int(name: str, default: str) -> int:
    """Parse an integer env var, falling back to `default` on blank/garbage.

    Mirrors _get_float above: these knobs (OPENROUTER_MAX_TOKENS,
    COSYVOICE_SAMPLE_RATE, JAITTS_SAMPLE_RATE, MUSETALK_SIZE/SPLIT_SIZE) are
    written by free-text config-panel fields, so a typo used to crash the
    pipeline at import with a bare ValueError instead of a clear warning.
    """
    raw = _get(name)
    if raw:
        try:
            return int(raw)
        except ValueError:
            import warnings

            warnings.warn(
                f"{name}={raw!r} is not an integer; using default {default}.",
                stacklevel=2,
            )
    return int(default)


@dataclass(frozen=True)
class Config:
    # --- language + targets ---
    language: str = _get("LANGUAGE", "en")  # "en" | "zh" | "th"
    ttfo_target_s: float = _get_float("TTFO_TARGET_SECONDS", "3")

    # --- product mode ---
    # ECHO_GUARD=1 mutes the mic while the bot is speaking (half-duplex). DEFAULT IS 0
    # (barge-in, mic always live; the P44 baseline) -> use headphones (or OS echo cancellation)
    # so the avatar's voice doesn't barge in on itself. The old steady-sync blocker is FIXED
    # (P53, 2026-07-15): the avatar client now holds TTSStoppedFrame until the turn's voice
    # fully drains, so BotStoppedSpeaking fires at true end of speech and the P11 stuck-mute
    # (mic dead after the first turn) can no longer arise. VERIFIED LIVE 2026-07-17 (the first
    # time =1 has ever run here): 3 driven turns under steady gave 3/3 mute->unmute cycles,
    # each unmute 1-2ms after BotStopped, and later turns still triggered -- the mic does not
    # stick. What is still UNJUDGED is whether half-duplex is WANTED: =1 kills barge-in for
    # the bot's whole reply, and replies here run 40-66s. Default stays 0 for that reason, not
    # because it's broken. docs/PROBLEMS-AND-FIXES.md P11/P53.
    echo_guard: bool = (_get("ECHO_GUARD", "0") or "0").lower() in ("1", "true", "yes", "on")

    # ALLOW_INTERRUPTIONS=1 (default) = the user can barge in and cut the bot off mid-reply
    # (pipecat broadcasts an interruption when the user starts speaking). =0 = the bot ALWAYS
    # finishes its turn; user speech during playback never cancels it ("can't interrupt").
    # This is the turn-start strategy's `enable_interruptions` flag -- NOT the mic mute
    # (that's echo_guard); this has no mute state machine at all, so the P11-era mute
    # concerns never applied to it. See main.py.
    allow_interruptions: bool = (_get("ALLOW_INTERRUPTIONS", "1") or "1").lower() in ("1", "true", "yes", "on")

    # --- VAD (Silero, local, always-on) — turn-taking feel vs. perceived latency ---
    # VAD_STOP_SECS is the big one: it is the silence the VAD must SEE before it calls
    # end-of-turn, so it sits ENTIRELY BEFORE the TTFO stopwatch starts (t0 = user-stopped)
    # -- it never shows up in a TTFO number, but the user waits every second of it. Lower =
    # snappier replies; too low and it cuts the user off mid-sentence (a pause between clauses
    # reads as end-of-turn). Raise it if the bot keeps interrupting; lower it to feel faster.
    # The other three are sensitivity: CONFIDENCE/MIN_VOLUME gate what counts as speech at all
    # (raise them on a noisy mic that keeps false-triggering), START_SECS is how much speech is
    # needed to call start-of-turn. Defaults = the values these had hardcoded in stages/vad.py.
    vad_stop_secs: float = _get_float("VAD_STOP_SECS", "0.5")
    vad_start_secs: float = _get_float("VAD_START_SECS", "0.2")
    vad_confidence: float = _get_float("VAD_CONFIDENCE", "0.7")
    vad_min_volume: float = _get_float("VAD_MIN_VOLUME", "0.6")

    # --- STT (Deepgram) ---
    deepgram_api_key: str | None = _get("DEEPGRAM_API_KEY")

    # --- STT provider switch (deliberate fallback switch, like TTS_PROVIDER) ---
    # deepgram = cloud streaming (default, interim partials);
    # sherpa   = local OFFLINE STREAMING (sherpa-onnx zipformer bilingual zh-en, CPU/~0 VRAM,
    #            drives turn-taking from its own ASR endpoint detector -- robust to a quiet mic).
    stt_provider: str = (_get("STT_PROVIDER", "deepgram") or "deepgram").lower()
    # sherpa: local streaming model dir + whether to convert zh output to Traditional (zh-TW).
    sherpa_model_dir: str = _get(
        "SHERPA_MODEL_DIR",
        "models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
    ) or "models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
    sherpa_traditional: bool = (_get("SHERPA_TRADITIONAL", "1") or "1").lower() in ("1", "true", "yes", "on")
    # How long a pause (seconds) ends your turn and FIRES the query to the LLM. Lower = snappier
    # (fires sooner after you stop), but too low can cut you off mid-sentence. Default 0.5.
    sherpa_endpoint_silence: float = _get_float("SHERPA_ENDPOINT_SILENCE", "0.5")
    # sensevoice: offline SenseVoice-Small (segmented via VAD + Smart Turn, GPU) -- a big
    # accuracy + noise-robustness upgrade over the 2023 zipformer. See sensevoice_stt.py.
    sensevoice_model_dir: str = _get(
        "SENSEVOICE_MODEL_DIR",
        "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17",
    ) or "models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
    # cuda = run the ~0.5GB model on the GPU headroom (RTF ~0.016); cpu = 0 GPU (also real-time).
    sensevoice_provider: str = (_get("SENSEVOICE_PROVIDER", "cuda") or "cuda").lower()
    sensevoice_traditional: bool = (_get("SENSEVOICE_TRADITIONAL", "1") or "1").lower() in ("1", "true", "yes", "on")
    # Endpoint trailing-silence (s) for SenseVoice's self-segmentation. LONGER than sherpa's 0.5
    # because SenseVoice is accurate on WHOLE utterances but garbles short fragments -- a short
    # value chops sentences at micro-pauses. 0.8 merges mid-sentence pauses into one segment.
    sensevoice_endpoint_silence: float = _get_float("SENSEVOICE_ENDPOINT_SILENCE", "0.8")

    # --- LLM (OpenRouter: one key, any model via OPENROUTER_MODEL) ---
    openrouter_api_key: str | None = _get("OPENROUTER_API_KEY")
    openrouter_base_url: str = _get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_model: str = _get("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite")
    # Pin OpenRouter to a specific backend provider (e.g. "Groq") so the LLM hop runs on a
    # low-latency inference host instead of the default transpacific Gemini route: cuts TTFT
    # ~1.1-1.6s (tail to 3.6s) -> ~0.7s tight, the dominant TTFO cost + all its variance.
    # Comma-list allowed. Empty = unpinned (today's behavior). See OPENROUTER_MODEL.
    openrouter_provider_only: str = _get("OPENROUTER_PROVIDER_ONLY", "") or ""
    # Hard cap on the reply length. This key sat in .env for weeks describing itself as a
    # "hard reply cap" while NOTHING read it -- so replies were in fact uncapped, and a
    # rambling 35s answer looked like a mystery latency bug instead of a missing knob.
    # 0/empty = unset (no cap), which is what the old behaviour actually was.
    openrouter_max_tokens: int = _get_int("OPENROUTER_MAX_TOKENS", "0")

    # --- LLM provider switch (deliberate fallback switch, like TTS_PROVIDER) ---
    # weather_chain = a dedicated Chinese weather bot backed by the NCU LangServe
    # endpoint; openrouter = the general-chat fallback. One flip reverts.
    llm_provider: str = (_get("LLM_PROVIDER", "openrouter") or "openrouter").lower()
    weather_chain_url: str = _get(
        "WEATHER_CHAIN_URL", "http://140.115.54.87:8000/chain/resWeatherChain"
    )  # base; the service appends /stream
    weather_chain_model: str = _get("WEATHER_CHAIN_MODEL", "gemma3:27b") or "gemma3:27b"
    # NCU serves the chain over IP-based HTTPS with a self-signed cert -> verify off.
    # Set WEATHER_CHAIN_VERIFY_TLS=1 for a properly-certed host (or the local mock).
    weather_chain_verify_tls: bool = (_get("WEATHER_CHAIN_VERIFY_TLS", "0") or "0") not in ("0", "false", "False", "")

    # --- Avatar memory harness (fully local: qwen2.5:3b on CPU via Ollama) ---
    # The chain is stateless, so the virtual human's growing memory lives here.
    # CPU-pinned (qwen2.5:3b-cpu) so MuseTalk + CosyVoice keep the whole GPU.
    avatar_memory: bool = (_get("AVATAR_MEMORY", "1") or "1").lower() in ("1", "true", "yes", "on")
    avatar_memory_dir: str = _get("AVATAR_MEMORY_DIR", "state/avatar_memory") or "state/avatar_memory"
    memory_llm_url: str = _get("MEMORY_LLM_URL", "http://localhost:11434/v1") or "http://localhost:11434/v1"
    memory_llm_model: str = _get("MEMORY_LLM_MODEL", "qwen2.5:3b-cpu") or "qwen2.5:3b-cpu"
    # Gated = only rewrite when the utterance looks context-dependent (keeps the
    # fast path fast; CPU rewrite ~0.77s when it does fire). 0 = always rewrite.
    memory_llm_gated: bool = (_get("MEMORY_LLM_GATED", "1") or "1").lower() in ("1", "true", "yes", "on")

    # --- TTS ---
    # cosyvoice (default) = the local CosyVoice streaming server, now IN THIS REPO at
    # tts/cosyvoice-server/ (run on vLLM in WSL). Female zero-shot voice, no per-token cloud cost.
    # jaitts = the local Thai server (CosyVoice cannot speak Thai).
    # (Removed 2026-07-14: the moss / elevenlabs / deepgram branches. Never selected, and an
    #  untried fallback is not a safety net -- it is code that rots. They are in git history.)
    tts_provider: str = (_get("TTS_PROVIDER", "cosyvoice") or "cosyvoice").lower()
    # CosyVoice local streaming server (local_services/cosyvoice_tts.py client ->
    # the in-repo server at tts/cosyvoice-server/). Native 24 kHz (Pipecat resamples down).
    cosyvoice_url: str = _get("COSYVOICE_URL", "http://localhost:8001")
    # (no cosyvoice_voice: the server ignores the per-request `voice` field -- it has ONE registered
    #  reference voice set by COSYVOICE_PROMPT_WAV/TEXT, swapped via the config panel's avatar presets.)
    cosyvoice_sample_rate: int = _get_int("COSYVOICE_SAMPLE_RATE", "24000")
    # JaiTTS-F5TTS local Thai server (local_services/jaitts_server/app.py, runs in the
    # shared F5 venv E:/f5-spike/.venv-f5). THE Thai voice path -- CosyVoice cannot speak
    # Thai. Same /tts/stream raw-PCM contract, so TTS_PROVIDER=jaitts reuses the CosyVoice
    # client pointed at JAITTS_URL. Voice = a fixed reference clip (JAITTS_REF); 24 kHz.
    jaitts_url: str = _get("JAITTS_URL", "http://localhost:8004")
    jaitts_sample_rate: int = _get_int("JAITTS_SAMPLE_RATE", "24000")

    # --- Avatar (local MuseTalk talking-head server on port 8002) ---
    # Force IPv4: the :8002 server binds 0.0.0.0 (IPv4 only), and on Windows a `localhost`
    # connect tries IPv6 ::1 FIRST and wastes ~2s failing over to IPv4 on every request. That
    # hit BOTH the build_avatar /health check AND the websockets.connect at connect time --
    # a flat ~2s+2s of Connect latency (measured 2026-07-18: urllib localhost 2031ms vs
    # 127.0.0.1 0ms). Normalizing here fixes it regardless of what AVATAR_URL says.
    avatar_url: str = _get("AVATAR_URL", "http://localhost:8002").replace(
        "//localhost:", "//127.0.0.1:")

    @property
    def is_mandarin(self) -> bool:
        return self.language.lower().startswith("zh")

    @property
    def is_thai(self) -> bool:
        return self.language.lower().startswith("th")

    @property
    def avatar_size(self) -> int:
        """Square output frame px (MUSETALK_SIZE, default 512). MUST equal the avatar
        server's size AND the transport's video_out_width/height in main.py -- a
        mismatch hands aiortc the wrong dims. Smaller = far less WAN bandwidth (the
        dominant lever vs jitter), at the cost of a softer face."""
        return _get_int("MUSETALK_SIZE", "512")

    @property
    def avatar_split(self) -> bool:
        """Split mode (MUSETALK_SPLIT): the avatar server streams only the mouth crop and
        /studio composites it over a pristine still. Default off (full-frame, /client works)."""
        return (_get("MUSETALK_SPLIT", "0") or "0").lower() in ("1", "true", "yes", "on")

    @property
    def avatar_split_size(self) -> int:
        """Fixed square px of the streamed mouth crop in split mode (MUSETALK_SPLIT_SIZE).
        MUST equal the avatar server's value; the WebRTC track is sized to it."""
        return _get_int("MUSETALK_SPLIT_SIZE", "256")

    @property
    def avatar_fps(self) -> float:
        """Output fps the avatar server pushes (MUSETALK_FPS, ~20 sustainable); main.py
        couples video_out_framerate to it (and avatar.py passes it to the client) so
        they can never diverge and drift."""
        return _get_float("MUSETALK_FPS", "20")

    @property
    def avatar_sync_with_audio(self) -> bool:
        """Whether the avatar pins video to audio (sync_with_audio + non-live transport).
        steady (default) = video-master => non-live transport (is_live=False), pins video
        to audio. live = audio-master => free-running transport (is_live=True). When on,
        main.py sets video_out_is_live=False so pipecat honors the per-frame sync."""
        mode = (_get("MUSETALK_SYNC_MODE", "steady") or "steady").lower()
        if mode not in ("steady", "prerender"):
            return False
        return (_get("MUSETALK_SYNC_WITH_AUDIO", "1") or "1").lower() in ("1", "true", "yes", "on")

    @property
    def system_prompt(self) -> str:
        if self.is_thai:
            return (
                "คุณเป็นผู้ช่วยด้วยเสียงที่เป็นมิตรและกระชับ "
                "ตอบเป็นภาษาไทยแบบภาษาพูดที่เป็นธรรมชาติ ประโยคสั้นๆ "
                "ตอบสั้นๆ ไม่เกิน 2-3 ประโยคเสมอ ถ้าเรื่องยาวให้ตอบสั้นๆ ก่อนแล้วถามว่าอยากฟังต่อไหม "
                "ห้ามใช้อิโมจิ บุลเล็ต หรือสัญลักษณ์จัดรูปแบบใดๆ เพราะข้อความจะถูกอ่านออกเสียง"
            )
        if self.is_mandarin:
            # First-sentence-short is a zh TTFO lever (KEPT): CosyVoice prefills the whole first
            # sentence before emitting any audio, so a short opener cuts the TTS first-chunk TTFB.
            # It caps only the OPENER, not the total length -- so a long, detailed answer can still
            # start fast. The 2-3-sentence brevity cap was removed so zh speaks fuller/longer.
            return (
                "你是一個友善、健談的語音助理。"
                "請用自然、口語化、適合朗讀的方式回答，內容要豐富、講得完整一些，"
                "每次回覆大約 5 到 8 句，把重點說清楚，適時補充細節、原因或例子，展開說明，不要只講一兩句就結束，"
                "第一句話要特別短（十個字以內），先點出重點，讓語音能馬上開始，後面幾句再慢慢展開，"
                "避免使用表情符號、條列符號或特殊格式。"
            )
        return (
            "You are a friendly, concise voice assistant. Answer in a natural, "
            "spoken style. Keep sentences short. Do not use emojis, bullet "
            "points, or any special formatting — your text will be read aloud. "
            "Keep every reply brief — at most 2-3 short sentences. If the topic is "
            "big, give the short answer and offer to say more, rather than monologuing."
        )


config = Config()
