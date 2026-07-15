# Research: getting TTFO under 1.5s (2026-07-15)

Current budget (fresh session, to the ear): LLM TTFB ~0.9s (Groq llama-4-scout)
+ TTS first-chunk ~0.85s (CosyVoice on vLLM) + lead-cushion fill (~14 frames @14fps,
fill-rate-bound) => 2.7-3.1s. Target < 1.5s. Web-swept last 12 months (agent run 2026-07-15).

## OUTCOME (end of 2026-07-15 session) — both engine swaps measured, both LOST to the tuned stack
- #1 Cerebras LLM: tail win only (no scout on OpenRouter; gpt-oss-120b 0.94s median vs Groq 1.08s
  but Groq spikes to 2.55s). **User skipped** — not worth changing the P21 model baseline.
- #3 vLLM-Omni TTS: 1.32s vs our 1.035s — per-request ref re-processing is structural. **Closed;
  track upstream for ref caching** (would be ~0.83s FLAT, stddev 0.02).
- #2 Eager end-of-turn: explained, **parked (user: not now)**. Still the strongest untouched lever
  — it's the only one that OVERLAPS two serialized costs (turn-confirm wait + LLM prefill) instead
  of shaving one, and it lives in the pre-t0 zone TTFO can't measure. Two shapes when revisited:
  Deepgram Flux (STT-stage rework, cloud) or a DIY speculative-prefill on nova-2 interims
  (fire LLM on sentence-final interim, cancel on TurnResumed-equivalent; own-aggregator change).
- Also still open and worth ~1s on turns 4+: the session-degradation bug (suspected reply-length
  backlog — the 35-38s replies; `OPENROUTER_MAX_TOKENS` was dead until 2026-07-14).
- Net: sub-1.5s will NOT come from a drop-in engine swap in July 2026; the stack is already at or
  ahead of the field per-stage. It comes from overlap (pre-t0), the degradation fix, and
  structurally a dedicated avatar GPU.

## Ranked: cheapest win first

### 1. Cerebras for the LLM hop — MEASURED 2026-07-15, verdict: tail win, not a median win. **USER SKIPPED** (2026-07-15): staying on Groq scout; the modest win didn't justify changing the P21 model baseline.
The research claim ("scout at TTFT 80-150ms") did NOT survive the probe:
- **Cerebras does NOT serve `llama-4-scout` on OpenRouter** (only deepinfra/groq/novita/google-vertex).
  The only zh-viable Cerebras-served model there is **`openai/gpt-oss-120b`**.
- Isolated probe from this box (scratchpad `llm_provider_ab.py` / `llm_ab2.py`, streaming,
  TTFT = first CONTENT token):
  - Groq scout (current baseline): median ~1.08s, range **0.58–2.55s** (2 of 6 runs > 1.8s — the variance is real)
  - Cerebras gpt-oss-120b, `reasoning effort=low`: median **~0.94s**, range 0.87–1.58s (tight)
- gpt-oss-120b zh quality: clean, natural TRADITIONAL zh, short openers — passes the eye.
- So the switch buys ~0.1–0.2s median + kills the 2.5s Groq tail spikes — worth having, but the
  ~0.9s TTFT floor from this box is network + prefill, not provider speed. NOT the −0.6s hoped.
- **Gotcha if adopted:** gpt-oss-120b is a reasoning model. `reasoning: {effort: "low"}` must be
  injected in `stages/llm.py` (same `Settings.extra` mechanism as `provider.only`) or the default
  effort buries the first content token behind reasoning tokens. `OPENROUTER_REASONING_EFFORT`
  in `.env` is currently a DEAD knob — it would need wiring.

### 2. Deepgram Flux — reclaims 200–600ms of PRE-t0 felt latency
Conversational STT (Oct 2025) with integrated end-of-turn (~260ms P50) + an "Eager End-of-Turn"
event: start LLM generation before the turn is confirmed, cancel if the user continues. Replaces
the separate Silero VAD stage. TTFO's t0 IS turn-end, so the metric can't see this win — but the
user's ear does (it stacks on top of any TTFO cut). Integration: new STT stage + eager-prefill
wiring; pipecat has turn-model precedent (Smart Turn v3 already runs here).
- https://developers.deepgram.com/docs/flux/voice-agent-eager-eot
- Papers: Speculative End-Turn Detector (arXiv 2503.23439), Next-Turn endpointing (arXiv 2606.18094)

### 3. vLLM-Omni CosyVoice3 — MEASURED 2026-07-15, verdict: NOT a win today (ref re-processing)
Spike ran the real thing (WSL `~/vllm-omni-spike/`, vllm 0.24.0 + vllm-omni 0.24.0, serving the
repo's own `Fun-CosyVoice3-0.5B-2512` copied to `~/models/` — /mnt/e torch.load hits 9P Errno-5).
Probe = `scratchpad/omni_ttfb_probe.py`, same 8 openers + leo ref as `_ttfb_variance.py`:
- **Current server (v2, leo, live conditions): median 1.035s** first chunk (0.68-1.45s, stddev 0.25)
- **vllm-omni (v3, leo 9.2s ref per-request): median 1.319s, stddev 0.023** — dead flat, length-independent
- Same probe with a 2.5s truncated ref: **0.83s median** → the fixed ~0.5s is PER-REQUEST REF
  PROCESSING (s3tokenizer/campplus on the clip). vllm-omni re-pays it on EVERY request even for an
  identical ref (no caching; CosyVoice3 support only landed in 0.24.0 — issue #1552 was stale);
  `ref_audio` is REQUIRED, no preset/registered-voice path in this build. Our server precomputes
  the zero-shot prompt ONCE at startup, so per-request cloning is structurally behind it.
- What IS attractive: the 0.023s stddev (vs our 0.25) shows the async two-stage pipeline is
  extremely steady. IF a ref-feature cache lands upstream (or we patch one in — the hash of the
  ref is constant per preset), realistic TTFB ≈ 0.8s flat with the full ref. That's ~0.2s + all
  the variance vs today. Track upstream; not worth carrying a fork for.
- Env gotchas paid (for a future retry): needs `s3tokenizer`; uv venv must use a MANAGED python
  (system 3.12 lacks Python.h); CC/CXX from the cosyvllm conda gcc; CUDA_HOME from the pip
  `nvidia-cuda-nvcc-cu13` wheel (`site-packages/nvidia/cu13`); uninstall flashinfer (JIT header
  mismatch on sm_120); model MUST live on WSL ext4, not /mnt/e.
- https://vllm.ai/blog/2026-06-23-vllm-omni-tts

### 4. TTS swap candidates (only if #3 stalls)
- **Chatterbox Multilingual v3** (Resemble, June 2026, MIT): dedicated Mandarin model (0.41% CER),
  ~150ms streaming first-packet, zero-shot cloning. CAVEAT: ~8GB VRAM — blows the shared-16GB
  budget (CosyVoice-on-vLLM currently uses ~2.3GB). Only viable with a second GPU.
- **Orpheus multilingual** (Llama-3B, vLLM-servable, ~200ms TTFB): now has a zh pack; zh quality
  unverified — judge by ear on Traditional input (T2S first, P43).
- IndexTTS-2.5, Fish/OpenAudio S1: zh + cloning rows, no verified sub-CosyVoice TTFB claims.
- Kokoro-82M rejected: no zero-shot cloning (Leo preset needs it).

### 5. Avatar cushion — mostly derivative, plus a CPU escape hatch
The 14-frame hold is CLOSED at 14 (user's eye) but its DURATION is fill-rate-bound: faster LLM
(#1) + faster TTS chunks shrink the hold without touching the lead. Beyond that:
- **OpenAvatarChat** (HumanAIGC, v0.4.1): open-source near-clone of this pipeline
  (VAD->ASR->LLM->TTS->MuseTalk/LiteAvatar), ~2.2s e2e — mine its MuseTalk handler + latency
  budgeting for ideas. https://github.com/HumanAIGC-Engineering/OpenAvatarChat
- **LiteAvatar** (HumanAIGC): audio2mouth at 30fps ON CPU — would end the vLLM<->MuseTalk GPU
  contention entirely. Open question: driving a custom preset face (Leo) vs their gallery.
- **Distilled few-step talking heads** (research horizon, needs sm_120 porting + eye check):
  TurboTalk 1-step (arXiv 2604.14580, 120x), LiveTalk 4-step (GAIR-NLP), REST (arXiv 2512.11229 —
  ID-context caching targets exactly the long-session degradation bug).

## Realistic path to <1.5s
Cerebras (0.9->~0.3) alone brings the chain to roughly LLM 0.3 + TTS 0.85 + fill ~0.4 ≈ 1.5-1.6s.
Add either a TTS-TTFB cut (#3) or Flux's pre-t0 reclaim (#2) and the felt latency lands clearly
under 1.5s. No pipeline rearchitecture required for any of the top-3.

## Constraint filter that killed most options
Traditional-zh quality + zero-shot cloning + shared-16GB VRAM + Windows/WSL install reality.
Every marketing latency number must survive the live eye/ear (P19/P33: a measured delta is not
a perceived one, in both directions).

---

# Appendix: the full research sweep (2026-07-15, web agent, window = last 12 months)

Everything surveyed at the start, including items that did NOT make the ranked list. Scope was
fixed by the user up front: keep the pipeline shape (VAD->STT->LLM->TTS->MuseTalk), find engines
that improve it — so speech-to-speech pipeline replacements were catalogued but out of scope.

## A. Streaming TTS engines (constraint: zero-shot cloning + zh)
| Engine | Claim | Fit verdict |
|---|---|---|
| Kyutai TTS / Unmute (2025) | ~220ms first audio; streams TEXT in too (could start synth before the first clause completes) | zh support weak; text-streaming idea is independently interesting |
| Chatterbox Multilingual v3 (Resemble, June 2026, MIT) | dedicated Mandarin model 0.41% CER, ~150ms streaming first packet, cloning | best zh contender on paper; **~8GB VRAM kills it on the shared card** |
| Orpheus TTS multilingual (Canopy) | Llama-3B backbone, ~200ms TTFB, vLLM-servable, zh pack now exists | zh quality unverified; 3B backbone heavier than CosyVoice's 0.5B |
| Kokoro-82M | near-instant first chunk | REJECTED: no zero-shot cloning (Leo preset needs it) |
| IndexTTS-2 / 2.5 (bilibili) | strong zh, cloning; 2.5 tech report benchmarks vs CosyVoice3 | no verified sub-CosyVoice streaming TTFB claim |
| Fish Speech / OpenAudio S1 | zh + cloning | same — quality-first, not latency-first |
| Spark-TTS / MegaTTS3 | zh-capable entrants | no streaming-latency edge documented |
| Higgs Audio V2/V3 (Boson, Apache-2.0) | 100+ langs, Qwen3 backbone, cloning | too heavy for 16GB-shared + sub-1s |
| VoxCPM (openbmb, arXiv 2509.24650) | tokenizer-free, context-aware cloning | emerging; watch |
| StreamMel (arXiv 2506.12570) | single-stage streaming, removes discrete-codec stage, SOTA latency claim | research-grade, no drop-in |
| Confucius4-TTS / Qwen3-TTS-1.7B / dots.tts / VibeVoice | recent zh entrants | mostly heavier than CosyVoice2-0.5B |
| vLLM-Omni CosyVoice3 | async chunked 2-stage serving | **MEASURED — see #3 above (lost on ref re-processing)** |

## B. Talking-head / lip-sync (attack the lead-cushion + GPU contention)
| Item | Claim | Fit verdict |
|---|---|---|
| OpenAvatarChat (HumanAIGC, v0.4.1) | open-source near-clone of THIS pipeline (VAD->ASR->LLM->TTS->MuseTalk/LiteAvatar), ~2.2s e2e, browser WebRTC | reference to MINE (handler architecture, latency budgeting), not rebuild. We're already at 2.7-3.1 with heavier constraints |
| LiteAvatar (HumanAIGC) | audio2mouth 30fps ON CPU | would END the vLLM<->MuseTalk GPU contention; open question: custom preset face (Leo) vs gallery styles |
| TurboTalk (arXiv 2604.14580) | 1-step distilled diffusion, 120x speedup, targets first-frame delay | research horizon; sm_120 porting + eye check needed |
| LiveTalk (GAIR-NLP, arXiv 2512.23576) | 4-step causal block-AR diffusion, 20x, streaming + identity preservation | same |
| REST (arXiv 2512.11229) | streaming + ID-Context caching for LONG streams | notable: targets exactly the session-degradation failure mode |
| Ditto motion-space (arXiv 2411.19509) | low first-frame latency | we removed Ditto 2026-06 (choppy/slow, memory: don't treat its sync as the bar); newer variants would need fresh evidence |
| HeyGem / Duix-Avatar | offline Windows-native digital human | file-gen oriented, not sub-second streaming |
| LivePortrait / LatentSync | quality-vs-latency poles around MuseTalk | no cushion win |
| Gaussian-splat heads (GaussianTalker etc.) | 100+fps after per-identity training | per-identity training conflicts with the preset-swap workflow |
| Wan-Streamer / StreamAvatar / MIDAS / InteractiveAvatar / Hallo-Live | end-to-end audio->video foundation models | where the field is heading; far too heavy for 16GB today — "watch" horizon |
| MuseTalk cushion re-work | hold is FILL-RATE-bound, lead=14 CLOSED | the real lever is faster upstream chunks, which #1/#3 were about |

## C. LLM hop + pre-t0 (turn-taking)
| Item | Claim | Fit verdict |
|---|---|---|
| Cerebras | "scout at 2000 tok/s, TTFT 80-150ms" | **MEASURED — see #1 above (marketing didn't survive OpenRouter+transpacific; tail win only; skipped)** |
| SambaNova | third fast-inference option | less latency-focused than the other two; untested |
| Deepgram Flux (Oct 2025) | conversational STT, integrated EOT ~260ms P50, EagerEndOfTurn event, replaces VAD; 200-600ms agent-response cut | **PARKED — strongest untouched lever** (see OUTCOME) |
| DIY speculative prefill | fire LLM on sentence-final nova-2 interim, cancel if speech resumes | the no-new-vendor shape of the same idea; papers: Speculative End-Turn Detector (2503.23439), Next-Turn endpointing (2606.18094) |
| Ultravox-class speech-in LLMs | delete the STT hop | changes the LLM (loses Groq scout); zh + latency unproven here |
| Qwen3-Omni / MiniCPM-o / GLM-4-Voice / Moshi / Step-Audio / Kimi-Audio, full-duplex research (DuplexOmni, SALMONN-omni, FlexDuo) | delete STT AND TTS hops | out of scope by user decision (rearchitects; loses MuseTalk PCM contract + voice cloning); strategically relevant only |

## Fields the agent added that shaped the verdicts
- **Pre-t0 reclaim** — TTFO's t0 IS turn-end, so endpoint latency is invisible to the metric but
  not the ear (this is why Flux ranks despite no TTFO delta).
- **Shared-GPU coexistence** — contention, not static VRAM, is the real cost (killed Chatterbox).
- **First-frame delay / NFE** — the axis the distillation papers optimize; maps to our cushion.
- **Long-session stability** — REST's ID-cache targets our degradation bug.
- **Windows/WSL install reality** — the vllm-omni spike burned ~6 layers of it (see #3 gotchas).

## Sources (agent run 2026-07-15)
TTS: kyutai.org/tts · resemble.ai (Chatterbox v3) · github.com/canopyai/Orpheus-TTS ·
vllm.ai/blog/2026-06-23-vllm-omni-tts + vllm-omni issue #1552 · arXiv 2601.03888 (IndexTTS 2.5) ·
arXiv 2506.12570 (StreamMel) · arXiv 2509.24650 (VoxCPM) · boson.ai (Higgs V3)
Avatar: github.com/HumanAIGC-Engineering/OpenAvatarChat · github.com/HumanAIGC/lite-avatar ·
github.com/duixcom/Duix-Avatar · arXiv 2604.14580 (TurboTalk) · arXiv 2512.23576 +
github.com/GAIR-NLP/LiveTalk · arXiv 2512.11229 (REST) · arXiv 2411.19509 (Ditto) ·
arXiv 2508.03457 (READ)
LLM/turn: developers.deepgram.com/docs/flux/voice-agent-eager-eot · livekit.com/blog (turn
detection) · arXiv 2503.23439 · arXiv 2606.18094 · github.com/QwenLM/Qwen2.5-Omni ·
arXiv 2606.09186 (DuplexOmni) · verticalapi.com/vs/groq-vs-cerebras
