# VisualLLm - Problems Found & How We Fixed Them - ARCHIVE

Retired sections from `PROBLEMS-AND-FIXES.md`: bugs whose **fix AND the code it
described no longer exist in the tree**, so the section is no longer the "why is
the code the way it is" reference for anything live. **Nothing here is current
truth** - it is the historical record (the paper's lessons draw on some of it, e.g.
P40's metrology). P-numbers are kept stable so a `(P11)` / `(P40)` reference
elsewhere in the repo still resolves here.

> Back to the live catalogue: `PROBLEMS-AND-FIXES.md`.

Why each was retired:
- **P11** - echo-guard stuck-mute. Root cause fixed at its source by **P53** (the
  `TTSStoppedFrame` is now held until the voice drains); the live knobs
  (`ECHO_GUARD=0` default, `=1` sound-but-unear-tested) live in `CLAUDE.md`. Kept
  here because P53's title is literally "(the P11 root cause)".
- **P13** - the MOSS-TTS provider. `TTS_PROVIDER=moss` and `local_services/moss_server/`
  were **removed 2026-07-14** (`pipeline/stages/tts.py` now raises on an unknown
  provider). The between-sentence-delay lesson (likely CPU contention, never
  confirmed) is moot with the engine gone.
- **P40** - the odd-byte "generic mouth" bug. Fixed at its single source by **P52**
  (producer-side carry in `run_tts`); the two consumer guards it added
  (`_srv_carry`, `_align_even`) are deleted. P52 carries the live mechanism; the full
  incident + the metrology lessons (a reference that shares the suspect input can
  never fail) are preserved below.

---

## P11 — With echo-guard on, voice stops triggering after the first turn (must type) ✅ FIXED (default flipped, 2026-06-23; **root cause fixed 2026-07-15 — see P53**)

> **2026-07-15 update:** the mechanism below is now FIXED at its root (P53): the avatar client holds the
> `TTSStoppedFrame` until the turn's voice fully drains, so `BotStoppedSpeaking` fires at true end of
> speech and the stuck-mute state machine can no longer arise.
>
> **2026-07-17 — the stuck-mute is now MEASURED dead, and this section is kept only as P53's subject.**
> `ECHO_GUARD=1` was run live under steady for the **first time since 2026-06-23** (the whole 42k-line
> log held **zero** mute events — the "pending ear-check" had simply never happened). Two proofs:
> 1. **Unit, both directions** (`archive/_tts_stop_order_test.py`): P53's hold ON → `last_audio@3,
>    stop@5` **PASS**; hold disabled → `last_audio@4, stop@1` **FAIL — the P11 order reproduces
>    exactly**. The test can fail, so its green is real (cf. the P1 `cudnn.benchmark` trap, where a
>    default faked a pass).
> 2. **Live, 3 driven turns under steady:** **3/3** `muted → unmuted` cycles, every unmute landing
>    **1–2 ms after** `Bot stopped speaking based on TTSStoppedFrame`, and **user turns kept firing
>    after bot turns** (23:46:59, 23:48:11). The symptom in the title — "stops triggering after the
>    first turn" — did not occur. (The run drove 3 and logged 2: loop 1 fell in the ICE handshake,
>    the known harness trap — the two real turns are 71.77 s apart, exactly the loop period. Size
>    `--btail` past the reply or echo-guard mutes a turn *legitimately* and fakes a P11 revival.)
>
> **So the default `0` no longer means "broken" — it means barge-in is the P44 baseline.** What is
> still unjudged is whether half-duplex is **wanted**: `=1` kills barge-in for the bot's entire reply,
> and replies measured **52 s and 66 s** that day. That is an ear/product call, not a probe's.

**Symptom.** Speaking a turn produced no response — the user had to type into the client for it to
work. Only after a bot turn; the first interaction could work, then the mic went dead.

**Root cause (a 3-way interaction, all pre-existing).** Echo-guard uses pipecat's
`AlwaysUserMuteStrategy`, which mutes the user on `BotStartedSpeakingFrame` and unmutes on
`BotStoppedSpeakingFrame`. (1) In **steady** sync the voice is held/released *late* (paced to video),
so the output transport sees audio arrive **after** the per-turn `TTSStoppedFrame` → it fires a
**second `BotStartedSpeaking`** right after the early unmute → re-mutes the user. (2) The screech fix
raised `BOT_VAD_STOP_FALLBACK_SECS` to 600 s, so the transport's audio-gap `BotStoppedSpeaking`
never fires afterward. Net: after a turn the mute state machine is left `_bot_speaking=True` with no
unmute → **mic stuck muted**, so STT gets no audio and no turn triggers. Typing bypasses the audio mute.
Confirmed in the log: `... user is now unmuted` (on TTSStopped) immediately followed by
`... user is now muted` while the avatar was still rendering the tail, with no unmute after.

**Fix.** Flip the default to **`ECHO_GUARD=0`** (barge-in; no mute strategy, mic always live) in
`pipeline/config.py`. Lowering `BOT_VAD_STOP_FALLBACK_SECS` was rejected (it reintroduces the P3
screech). Verified: with `ECHO_GUARD=0` a synthesized voice turn triggered cleanly
(`User started speaking` → LLM → `TTFO {count:1, pass:True}`) with **no** mute events. Tradeoff: the
mic is always live → use headphones / OS echo cancellation.

> **The `live`-only restriction this paragraph used to state is RETIRED (2026-07-17).** It read
> "`ECHO_GUARD=1` remains valid **only** with `MUSETALK_SYNC_MODE=live`", and proposed a
> `TTSStarted`/`TTSStopped` mute as the future fix. P53 fixed the ordering itself, so no such
> rework is needed and `=1` is valid under steady — **measured, see the banner above.** The claim
> had also leaked into `pipeline/config.py` and `pipeline/stages/stt.py` (both corrected).

---

## P13 — MOSS-TTS "delay between sentences" ⚠️ NOT RESOLVED (2026-06-29; streaming+eager helped TTFB but the felt latency is still bad — and got WORSE)

> **HONEST STATUS (2026-06-29, end of session).** The streaming + eager changes below cut the *isolated*
> per-request TTFB (benchmarked ~0.4 s), but **the user reports the between-sentence delay is still there
> and overall latency is now WORSE.** So the isolated-TTFB win did NOT translate to a smooth live
> conversation — do not trust the "fixed" framing. **Leading hypothesis (untested): CPU contention.**
> This session also moved the **LLM onto a CPU-pinned local Ollama** and was running the memory harness,
> the memory-sim, and the weather mock — all on CPU — while the GPU ran CosyVoice-vLLM + MuseTalk. The
> original smooth baseline used a **cloud** LLM, leaving the CPU free. The most likely culprit is the
> machine being CPU-saturated, not the TTS engine. **Plan: revert `.env` to the baseline (cloud LLM +
> CosyVoice) next session and re-measure end-to-end TTFO before touching MOSS again.** The vLLM-Omni path
> (no torch.compile, GPU-served) remains the real fix for MOSS if it's pursued.

**Symptom.** With `TTS_PROVIDER=moss`, the avatar took a long beat before each sentence — felt like a
big lag, "is it the bigger model?". (Still present after the changes below.)

**Root cause (measured, not guessed).** Two separate things, neither the parameter count:
1. **The first server was non-streaming.** It called MOSS's `inferencer.generate()` (whole sentence),
   THEN streamed the finished PCM. Measured: time-to-first-audio **8.55 s** ≈ total wall **8.58 s** —
   they were identical, i.e. the avatar waited for the entire sentence before any sound. (The 1.7B size
   only makes steady-state RTF ~1.6, a minor factor.)
2. **Once streaming, `torch.compile` recompiled per sentence-length.** The streaming rewrite
   (`MossTTSRealtimeStreamingSession` + `AudioStreamDecoder`: push_text → decode → drain → flush) dropped
   warm TTFB to ~0.4 s — but the **first** time it saw each new token-length it recompiled **3–40 s**, and
   a real reply has many lengths, so the spikes landed *between sentences*.

**The fix.** (a) Stream the first chunk (the rewrite above). (b) Run **eager** — the server defaults
`TORCHDYNAMO_DISABLE=1` (override `MOSS_COMPILE=1`). Eager has **no recompiles**: every sentence is a
consistent **~0.35–0.5 s** TTFB (vs compiled's 0.25 s warm but 3–40 s spikes). Verified across varied
lengths: worst-case TTFB 0.53 s, zero spikes. Tradeoff: eager's long-sentence steady-state is a bit
slower; the both-fast-and-no-spikes path is **vLLM-Omni** (MOSS supports it natively — the next step).

**Install gotchas hit along the way** (all in the server docstring): MOSS's streaming codec path needs a
**C compiler** for triton (`CC`/`CXX` → `conda install -c conda-forge gcc gxx`); `torchcodec` couldn't
`dlopen` until **ffmpeg pinned to 7.1** (8.x is too new) + **`nvidia-npp-cu12`** installed +
**`LD_LIBRARY_PATH`** covering torch/lib + the env lib + the `nvidia/*` pip libs. Files:
`local_services/moss_server/app.py`.

---

## P40 — THE avatar "same generic mouth pattern" bug: the lip-sync model was fed NOISE (odd-byte misalignment) ✅ SHIPPED (2026-07-10, 16th session)

**Symptom (the multi-session "live lipsync is bad, offline is good" problem — the REAL one).** The live avatar's mouth
opened and closed in roughly the same wordless pattern regardless of what was said, and never rested during pauses. The
user was explicit, twice: *"it is not an A/V sync problem"* and *"the avatar keeps moving the same kind of mouth pattern."*
The voice itself always sounded perfect.

**Root cause.** Audio is int16: **two bytes per sample**. The TTS hands over `TTSAudioRawFrame` buffers with **odd byte
counts** (CosyVoice's `iter_chunked` splits mid-sample). Two paths consume that audio:

- **Downstream (what you HEAR)** — `_align_even()` **carries** the dangling odd byte into the next frame so the PCM stays
  whole-sample. This is the old P3 anti-screech fix. The voice is therefore always clean.
- **Avatar-bound (what the mouth is generated from)** — `_to_16k_mono_pcm()` **DROPPED** it: `if len(audio)&1: audio = audio[:-1]`.

Drop one byte and the *next* buffer begins half a sample late. Every int16 after it is assembled from the **low byte of one
sample and the high byte of the next** -> loud, constant, full-band noise. MuseTalk lip-syncs off a **Whisper of the
waveform** (not the text, cf. P18/P33), so Whisper heard static: no words, no consonants, **no pauses** -> the UNet produced
a continuous generic flap. **Voice clean, mouth garbage** — and completely independent of fps, `MUSETALK_SIZE`, GPU
contention and A/V sync, which is why every earlier theory failed against it.

**Evidence (same turn: `_delivered_voice.wav` 24k = what you hear, vs `_live_turn_pcm.wav` 16k = what the avatar gets):**

| metric | aligned (correct) | 1-byte misaligned | ACTUAL avatar-bound | white noise |
|---|---|---|---|---|
| peak xcorr vs the true voice (any lag +-2s) | 1.0 | — | **0.008** | ~0 |
| zero-crossing rate | 0.227 | 0.459 | **0.453** | ~0.5 |
| envelope dynamic range | 8.8 dB | 0.9 dB | **0.4 dB** | ~0 |
| quiet-10th-pct / peak (are there pauses?) | 0.006 | 0.517 | **0.349** | ~1 |
| RMS | 4000 | 13876 | **13531** | — |

A deliberate 1-byte shift of the true voice reproduces the avatar-bound signature almost exactly. The spectrum is the
clincher: real speech is `-0.8 dB @0-1k` rolling off to `-38.5 dB @7-8k`; the avatar-bound audio is **flat**
(`-8.3 / -5.8 / -6.2 / -6.2 / -9.5 dB`) — the signature of noise, not a voice.

**FIX (`musetalk_video.py`, the `TTSAudioRawFrame` branch).** Carry the remainder across chunks, exactly as `_align_even`
does downstream, aligned to a whole sample-FRAME (`2 * channels`, so stereo stays correct):

```python
stride = 2 * ch
data = self._srv_carry + frame.audio
keep = len(data) - (len(data) % stride)
self._srv_carry = data[keep:]          # hold the remainder for the next chunk
data = data[:keep]
pcm = _to_16k_mono_pcm(data, sr, ch) if data else b""
```

`self._srv_carry` is initialised in `__init__` and reset in `_reset_turn()`.

> **UPDATE 2026-07-15 (P52): `_srv_carry` is REMOVED.** The odd buffers it carried are now fixed at
> their single source (`cosyvoice_tts.py::run_tts` producer-side alignment — see P52), so frames reach
> this branch whole-sample. `_to_16k_mono_pcm`'s odd-drop remains as a crash guard only and should
> never fire; if it ever does, fix the PRODUCER (a dropped byte here is exactly this bug again).

**Verified live** (drove a real turn with `MUSETALK_DUMP_PCM=1`, compared the avatar-bound PCM against `resample_poly` of
the delivered 24k voice): peak xcorr **0.008 -> 0.969 @ lag 0.000s**; ZCR 0.453 -> **0.185** (target 0.189); envelope dyn
range 0.4 -> **8.8 dB** (target 8.8); quiet/peak 0.349 -> **0.040** (target 0.043); RMS 13531 -> **4089** (target 4113); the
spectrum became speech-shaped. Turn healthy: `audio 19.04s video 19.08s (229 frames) end drift +-0.04s`. Watchable live
output: `output/FIXED_live_avatar.mp4`.

**RESIDUAL (open, minor).** xcorr is 0.969, not 1.0, because `_to_16k_mono_pcm` still downsamples 24k->16k with bare
`np.interp` and **no anti-aliasing low-pass**, folding everything above 8 kHz back into the speech band (that band carries
the consonants). Fix with a streaming `resample_poly` if the visemes still look soft. This is "words slightly smeared", not
"static" — a different order of magnitude from the bug above.

### ⚠️ METROLOGY — how this stayed hidden for THREE sessions (read before writing another avatar probe)

1. **"Delivered frames == offline render, byte-identical" proves only that the RENDER IS DETERMINISTIC.** It cannot catch a
   corrupt *input*, because the offline side was fed **the same corrupt PCM** (`_live_turn_pcm.wav`). A test whose reference
   shares the suspect input can never fail. **Always ask what the reference is actually fed.**
2. **The "good" offline reference was accidentally repaired audio.** Earlier prerenders were fed a voice captured off
   **WebRTC** — i.e. the *downstream* copy, which `_align_even` had already fixed. So the reference bypassed the broken code
   path and always looked great. The user cracked the case by noticing *"offline is bad now too"* — the first time the
   offline render had been fed the audio the avatar actually receives.
3. **Mouth-motion vs audio-RMS correlation is USELESS** (4th time it misled). It scored the noise-driven render `0.97x` and
   the correct one `1.07x`; it previously scored a 1:1 offline reference as `+280 ms` out of sync. Energy != phoneme shape.
4. **Never verify A/V sync from a WebRTC capture reconstructed by ARRIVAL time.** Under `steady`, audio is released in
   bursts paced to the render (`hold` can exceed 30 s), so arrival-order reconstruction distorts the A/V relationship and
   the metric measures the harness, not the system. Use the uncompressed `MUSETALK_DUMP_DELIVERED` dump.
5. **Dead theories, all disproven by the above** — do not re-open without new evidence: fps/OOD-Whisper-stride
   (`MUSETALK_SIZE 512->256` to reach 25 fps), held-frame stalls, shared-GPU contention, VP8/transport, segmentation/Whisper
   context. The render was correct the whole time; it was being *told* the wrong thing.

---

