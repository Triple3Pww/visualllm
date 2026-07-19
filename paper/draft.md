---
title: "VisualLLm: A Fully Local Streaming Architecture for Real-Time Multilingual Talking-Head Agents on a Single Consumer GPU"
author: "[AUTHORS — user to supply names, order, affiliations]"
bibliography: references.bib
---

# Abstract

We present VisualLLm, a real-time conversational agent that turns a user's speech into a
photoreal talking-head answer — speech recognition, language model, speech synthesis, and
lip-synchronized video — running fully streaming and fully local on a single consumer GPU. On
an RTX 5060 Ti (16 GB), the system reaches a median time-to-first-output of 2.20 s (English)
and 2.92 s (Mandarin) over fresh-session trials, with the whole stack resident in ~5.6 GB of
GPU memory, and supports multiple languages and swappable avatar identities. The contribution
is architectural rather than model-level: a linear streaming pipeline in which every stage
overlaps its successor; per-language latency engineering grounded in a measurement methodology
that attributes every millisecond from the end of the user's utterance to the user's ear; and
an audio–visual synchronization design in which each rendered frame declares the audio
position it covers, making lip drift structurally impossible rather than heuristically
corrected. We report the measured cost of each design decision, the engineering lessons —
including two systematic ways latency instruments and human perception disagree — and the
limitations of the current system.

# 1. Introduction

Embodied conversational agents — systems you *talk to* and that answer as a talking face —
are commercially available today as cloud services, and as research prototypes at the model
level. Between the two lies a gap this paper addresses: what does it take, architecturally,
to run the entire loop — speech in, photoreal lip-synchronized speech-and-video out — on one
consumer GPU, with no per-minute API bill, no audio or video leaving the machine, and latency
low enough for conversation?

The latency question is the sharp one. We adopt three seconds from end-of-utterance to the
first synchronized audio-and-video as the design budget — beyond that, an exchange stops
feeling like dialogue — and a naive serial composition of even good local components
(transcribe, then generate, then synthesize, then render) spends that budget several times
over. And local operation on a single card creates a problem cloud
architectures never face: the speech synthesizer and the video renderer *contend for the same
GPU*, so the renderer cannot promise a frame rate, and audio–visual synchronization becomes a
first-class design problem rather than a transport detail.

This paper describes VisualLLm, a system built to answer that question, and reports what it
measures. The design principles are: stream everything (the language model's first sentence
is being spoken before the answer is finished being generated, §3); spend engineering where
the measured milliseconds are (a to-the-ear latency waterfall attributes every stage, §4);
and make synchronization structural (every rendered frame carries the renderer's own account
of the audio it covers, §5).

Concretely, the contributions are:

1. **An end-to-end streaming architecture** for a fully local, multilingual talking-head
   agent on one consumer GPU, in which every stage is a configuration-selected single
   provider with a local alternative (§3).
2. **Per-language time-to-first-output engineering** and the measurement methodology that
   drives it: a same-clock waterfall from end-of-utterance to the user's ear, and levers
   whose measured effects are reported individually (§4, Table 2).
3. **A drift-free audio–visual synchronization mechanism**: video-master pacing driven by
   per-frame audio-position metadata declared by the renderer, verified by an exact
   per-turn invariant (§5).
4. **Resource engineering** that fits the TTS language model (on vLLM) and the TensorRT
   renderer in ~5.6 GB of shared VRAM (§3.4, §6.4).

We evaluate on the deployment hardware with 10 fresh-session trials per language (§6):
median time-to-first-output 2.92 s (Mandarin) and 2.20 s (English), ~5.6 GB resident, 12 fps
delivered video, end-of-turn A/V drift within ±0.04 s. Section 7 reports the engineering
lessons — including where automated latency probes and human perception systematically
disagree — and the system's honest limitations.

# 2. Related Work

**Talking-head generation.** Audio-driven talking-head synthesis has progressed from
GAN-based mouth-region dubbing (Wav2Lip [@wav2lip2020]) to latent-space and diffusion methods.
MuseTalk [@musetalk2024] inpaints the lower face in a VAE latent space with a multi-scale U-Net,
generating 256×256 mouth regions above 30 fps; Ditto [@ditto2024] drives a photorealistic
renderer through an explicit motion-space diffusion model with streaming inference. These works
optimize the *model*: they report offline or single-stream throughput, and leave open the
systems questions a conversational agent raises — end-to-end latency from the user's utterance,
synchronization of generated video with separately streamed speech, and coexistence with the
other models (LLM, TTS) competing for the same GPU. We build on MuseTalk as the rendering model
and address exactly those questions.

**Commercial real-time avatar APIs.** Hosted services such as Simli [@simli] and HeyGen's
Interactive Avatar [@heygen] deliver conversational avatars over WebRTC, but as cloud products:
per-minute pricing, provider-controlled identity models, and audio/video that must leave the
user's machine. Our first prototype used such a service; the architecture presented here
replaces every hosted stage with a local one on consumer hardware, which is the deployment
regime — cost-free at the margin, private, and offline-capable — that this paper targets.

**Open conversational-avatar systems.** OpenAvatarChat [@openavatarchat] is the closest open
system: a modular ASR–LLM–TTS–avatar pipeline on a single PC, reporting ~2.2 s average response
delay with its lightweight avatars. Its MuseTalk mode couples audio *bytes* into each video
frame packet, which prevents drift by construction but ties the delivered voice to the 16 kHz
lip-sync copy; we adopt the per-frame coupling idea as *metadata* instead (§5), keeping the
full-quality TTS audio. Orchestration frameworks — Pipecat [@pipecat], on which our pipeline is
built, and LiveKit Agents [@livekitAgents] — provide the streaming frame transport and turn
management substrate but no avatar, no latency budget, and no A/V-sync policy; those are the
contributions of this work.

**Streaming TTS.** CosyVoice 2/3 [@cosyvoice2_2024; @cosyvoice3_2025] generate speech tokens
autoregressively with an LLM backbone, enabling first-audio latencies well under a second when
the backbone is served efficiently; we run it on vLLM [@vllm2023] and engineer the input side
(§4) so that first-chunk cost, which scales with input sentence length, stays off the critical
path.

To our knowledge no published system combines fully local operation, a single consumer GPU,
multilingual support, and a structural (rather than best-effort) audio–visual synchronization
guarantee at the ~3 s time-to-first-output level we report.

# 3. System Architecture

![Fig. 1. System architecture. Solid boxes are processes/stages; the dashed region shares one
consumer GPU. Every stage is selected by environment configuration and has a local
alternative.](figures/architecture.svg)

## 3.1 A linear streaming pipeline

VisualLLm is organized as a single linear pipeline (Fig. 1) built on the Pipecat framework
[@pipecat]: browser microphone → WebRTC transport → voice-activity detection (Silero VAD
[@sileroVad]) → streaming STT → LLM → streaming TTS → avatar → WebRTC transport → browser.
Turn-taking is decided by a semantic turn analyzer (Smart Turn [@smartturn]) running on the
VAD's speech segmentation, so end-of-turn does not wait for a fixed silence timeout.

The defining property is that *every* stage streams and overlaps with its successor. The LLM's
tokens are aggregated into sentences and flushed to TTS as soon as the first sentence is
complete — synthesis of sentence 1 begins while the LLM is still generating the rest of the
answer. TTS emits audio in chunks as they are synthesized, and each chunk is forwarded to the
avatar renderer the moment it arrives. As a result the user hears (and sees) the beginning of
the answer while most of it does not yet exist; the pipeline's time-to-first-output is set by
the *first* sentence's path, not the answer's length (§4).

## 3.2 One provider per stage, local alternatives everywhere

Table 1 lists the stages. Each is constructed by a thin factory that instantiates exactly one
provider chosen by environment configuration — deliberate fallback *switches*, not runtime
multi-provider branching, and an unknown provider name raises at startup rather than silently
substituting a cloud service. The default configuration uses one cloud dependency (STT) and one
cloud-or-local dependency (LLM); both have drop-in local alternatives, so the whole system can
run with no network access at all.

**Table 1. Stages and providers.**

| Stage | Default | Local alternative | Runs on |
|---|---|---|---|
| VAD | Silero VAD [@sileroVad] | (already local) | CPU |
| Turn end | Smart Turn v3 [@smartturn] | (already local) | CPU |
| STT | Deepgram nova-2 streaming (en/zh/th) | sherpa-onnx streaming zipformer, bilingual zh-en, zh→Traditional via OpenCC [@sherpaOnnx; @opencc] | cloud / CPU |
| LLM | OpenAI-compatible endpoint (OpenRouter, `llama-4-scout`) | same interface pointed at a local Ollama server | cloud / GPU |
| TTS | CosyVoice2-0.5B [@cosyvoice2_2024], LM served by vLLM [@vllm2023] | (already local; CosyVoice 3 [@cosyvoice3_2025] selectable) | GPU (WSL2) |
| Avatar | MuseTalk [@musetalk2024] + TensorRT [@tensorrt] | (already local) | GPU |
| Transport | WebRTC (aiortc [@aiortc]) | — | CPU |

## 3.3 The avatar as a separate GPU process

The renderer runs as its own process — a websocket server in its own Python environment —
rather than inside the pipeline, isolating a heavyweight GPU workload (and its CUDA/TensorRT
state) from the latency-critical asyncio event loop that carries WebRTC and the pipeline. The
wire contract is small: the client streams the TTS audio down-sampled to 16 kHz mono PCM; the
server returns rendered RGB frames at a steady frame rate plus `video_start` / `video_clock` /
`video_end` markers counting only genuinely rendered frames. Each binary frame carries a
16-byte header declaring what it is (a real render, a held re-send, or an idle frame) and how
much of the turn's audio it covers — the basis of the synchronization design in §5. The
delivered voice is *not* the 16 kHz lip-sync copy: the original 24 kHz TTS audio goes to the
browser, and the header carries positions, not bytes.

MuseTalk derives its lip motion from a Whisper [@whisper2022] encoding of the incoming
waveform and inpaints the mouth region of a fixed portrait, so an avatar identity is data, not
code: a portrait image plus a cloned voice reference (CosyVoice zero-shot) and a language
setting. Swapping identities is a configuration change.

## 3.4 Sharing one consumer GPU

TTS and renderer share a single 16 GB consumer GPU (RTX 5060 Ti). Two measures make this
coexistence reliable. First, vLLM's KV-cache pool is sized to the actual workload: the TTS LM
serves one sentence per request, so a pool sized for ~7 maximum-length sequences (7% of GPU
memory) replaces the default multi-gigabyte reservation with ~0.16 GiB, without measurable
speed cost. Second, once the renderer's TensorRT engines are loaded, the PyTorch copies of its
UNet and VAE are freed (−1.8 GB): the TRT-vs-PyTorch fallback decision happens at load time,
after which the originals are dead weight. The project's processes hold ~5.6 GB, leaving most
of the card free (Table 4). The launcher starts the TTS server before the renderer; with the
reduced pool this ordering is defensive rather than required, as the inference server's memory
budget is a fixed fraction of the card and is not charged for other processes' allocations.

## 3.5 Client

The browser client is a plain static page speaking standard WebRTC signaling; no build step.
An optional *split mode* streams only a fixed 256×256 mouth crop over the video track and
composites it client-side over a losslessly transmitted background still — concentrating the
video codec's bitrate budget on the only region that changes. A typed-input path and a
transcript view ride the same connection as side endpoints, leaving the pipeline structure
untouched.

# 4. Latency Engineering

![Fig. 2. Per-stage latency to the user's ear, median stage deltas over 10 fresh sessions per
language. The LLM segment is near zero in this probe because the synthetic microphone drive
lets generation overlap the tail of the recorded utterance; on real turns the LLM hop is
~0.7–0.9 s (see §4.3). Bar totals are sums of independently computed stage medians; median
TTFO is reported in Table 3.](figures/waterfall.png)

## 4.1 The metric and how it is measured

Our primary metric is **time-to-first-output (TTFO)**: the interval from the instant the turn
analyzer declares the user's utterance finished (t0) to the instant the synchronized
audio+video answer starts toward the client. We additionally measure the remaining path to the
user's ear with a headless WebRTC probe that runs on the same machine as the pipeline — the
pipeline log's t0 and the probe's packet-arrival clock are then the *same wall clock*, so
per-stage log anchors and client-side arrival times stitch into one waterfall that sums to a
true end-to-end number (Fig. 2). Only the final segment (browser jitter buffer and playout,
~0.15 s) is an estimate rather than a measurement. What TTFO deliberately does *not* include is
the turn-end detection time before t0: voice-activity and semantic turn-end decisions are
latency the user feels but that no post-t0 engineering can recover, and we report them as out
of scope.

## 4.2 Where the time goes

At the measured baseline (Table 3: median TTFO 2.92 s zh, 2.20 s en), the two dominant post-t0
costs are the TTS first chunk (median 0.89 s in both languages) and the synchronized-start
lead-hold (median 0.60–0.64 s; §5) — the LLM hop, transport, and playout make up the rest.
This profile is the *result* of the engineering below; at the project's start the same path
cost 7–8 s on bad turns.

## 4.3 The levers

**Serve the TTS language model properly.** CosyVoice's autoregressive LM originally ran in
plain PyTorch, costing ~3.4 s to first audio. Moving that model onto vLLM [@vllm2023] (with
CUDA graphs on) cut first-chunk latency to ~1.1 s — the single largest win, and the one that
turned the avatar's lip-start lag from a defect into a tuning problem. One subtlety: vLLM's
serving optimizations silently dropped CosyVoice's repetition-aware sampling, which
intermittently looped Chinese synthesis on the silence token (a 4 s sentence becoming 12 s of
dead air); it was restored as a custom logits processor.

**Send the TTS a clause, not a paragraph.** CosyVoice prefills its entire input sentence
before emitting the first audio token, so first-chunk cost scales with *input* length — a
16-word opening sentence cost ~3.0 s where a short clause costs ~1.7 s. The pipeline therefore
flushes the turn's first *clause* to TTS early. The split is language-specific. For English,
split at a comma or word boundary within an 18–32 character window (measured: TTFO ~4.6 →
~3.2 s). For Chinese — where the English rule never fires (no ASCII commas, no spaces) — split
only at full-width punctuation (，；：), never at a character cap, because a cap cuts inside
words; with a 5-character minimum so the first clause's audio covers the next clause's
synthesis (measured on long-opener turns: 4.78 → 3.08 s, with 59–65 ms audio gaps — no audible
pause).

**Pin the LLM route.** The LLM hop is a cloud round-trip with provider-dependent variance;
routed by default, it cost 1.6 s median (zh) with a 7–8 s tail. Pinning the OpenAI-compatible
router to a single fast inference provider cut it to 0.80 s (zh) / 0.67 s (en) median on real
turns and eliminated the tail — configuration, not code. (Fig. 2 shows this row near zero
because the probe's recorded utterance lets generation overlap the end of "speech"; the
real-turn numbers are the honest ones.)

**Accept the lead-hold.** The remaining large segment, the ~0.6 s synchronized-start cushion,
is §5's deliberate tradeoff: 14 lip frames must exist before the voice starts. Every attempt
to shrink it below 14 frames passed the automated probe and failed human viewing (delay or
visible freezes) — it is the price of a face that is moving when the voice begins, and we
report it as such rather than tuning it away on instrument evidence (§7).

**Table 2. Levers and their measured effect (development-time A/B measurements).**

| Lever | Before → after |
|---|---|
| TTS LM on vLLM (vs PyTorch serving) | first chunk ~3.4 → ~1.1 s |
| First-clause flush, en (18–32 char window) | TTFO ~4.6 → ~3.2 s |
| First-clause flush, zh (full-width punctuation only) | long-opener turns 4.78 → 3.08 s |
| LLM provider pin | LLM hop 1.64 → 0.80 s median (zh), tail eliminated |
| TensorRT render path (enables §5's video-master sync) | per-segment render 455 → 171 ms (contended headroom 1.04× → 2.05×) |

# 5. Audio–Visual Synchronization

![Fig. 3. Proto-2 coupling under steady (video-master) sync. Every frame declares its kind and
the cumulative audio position it covers; the client releases voice only up to the position of
the last real frame shown, so a render stall pauses the voice instead of letting it drift
ahead.](figures/proto2_sync.svg)

## 5.1 Why synchronization is the hard problem

TTS produces audio faster than real time, and WebRTC will happily deliver it immediately. The
renderer, however, shares its GPU with the TTS language model and cannot guarantee its frame
rate under contention. If audio is the master (`live` mode in our system), the voice starts
instantly and the lips trail by whatever the renderer is behind — under load, visibly. If video
is the master, the voice must be paced to the frames that were *actually rendered*; done
naively — pairing audio to frames by index arithmetic (`frame i` covers `i/fps` seconds) — the
mapping silently breaks whenever the server and client disagree about the frame rate, or when
the server re-sends a held frame to keep the picture alive during a stall. We shipped and then
removed exactly such a heuristic layer: byte-comparing frames to *guess* whether one was a held
re-send, and index arithmetic that an fps misconfiguration could shift without any error
surfacing.

## 5.2 The frame declares itself

Our current design (Fig. 3) makes the coupling explicit. When the client requests protocol
version 2 at connection time (acknowledged by the server; older peers keep the bare-frame wire
format), every binary frame is prefixed with a 16-byte header: a magic tag, a *kind* byte —
0 = real render, 1 = held re-send, 2 = idle — and a 64-bit *audio position*: the cumulative
count of real 16 kHz samples of the current turn that are covered once this frame is shown.

The steady-mode client then implements one rule: release buffered voice up to the audio
position of the last real frame displayed. Held and idle frames are declared, not guessed, and
advance nothing. Because the position is the renderer's *own account* of what it has rendered,
a frame-rate mismatch between the processes structurally cannot shift the audio-to-lip mapping
— the failure mode is gone, not merely handled. This translates OpenAvatarChat's
audio-bytes-in-packet coupling [@openavatarchat] into metadata: coupling bytes would tie the
delivered voice to the renderer's 16 kHz working copy, whereas positions let the browser
receive the original 24 kHz TTS audio.

Two consequences follow. A synced *start*: the voice is held until a small cushion of lip
frames (14 at 12 fps) is rendered, so speech never begins on a frozen face — this cushion is a
deliberate latency cost, visible in §4's waterfall, and doubles as the shock absorber for
mid-turn render hiccups. And a graceful *stall*: if the renderer falls behind, the voice pauses
and resumes rather than drifting out of sync (the `live` fallback mode inverts the tradeoff:
instant voice, best-effort lips).

## 5.3 Keeping the renderer at frame rate

Video-master sync is only viable if stalls are rare, which on a shared GPU means the render
path must hold its budget *under TTS contention*. Porting MuseTalk's UNet and VAE to TensorRT
[@tensorrt] cut the per-segment render cost from ~455 ms to ~171 ms (8-frame segments) at the
deployed 512 px configuration, raising headroom against the 667 ms real-time budget from 1.5×
to 3.9×, and from 1.04× to 2.05× under sustained TTS contention.

What the port buys is *margin*, and how that margin must be measured is itself a result. At the
deployed 12 fps the PyTorch path also clears the budget — by 4 % — so both paths report the
same flat +0.36 s offset, and an ablation run at the deployed frame rate alone would wrongly
conclude the port was inert. Tightening the budget to 25 fps separates them immediately: 13.6 s
of speech accumulates 4.04 s of held-frame padding on the PyTorch path (101 held frames) versus
0.32 s on TensorRT (8), which video-master would surface as seconds of paused voice. The
general point, developed in §7: an ablation that holds the operating point fixed measures
whether a component is *currently binding*, not whether it is load-bearing. Two non-obvious
details mattered: cuDNN autotuning (`cudnn.benchmark`) had to be disabled because
the turn-start segment's distinct tensor shape triggered a ~16 s re-autotune spike on the first
segment of *every* turn; and per-segment frame counts use ceiling (not floor) sizing so a frame
rate that does not divide the audio rate cannot systematically shorten the video against the
audio.

## 5.4 Verifying synchronization

The header also makes sync *testable*: at end of turn, the last frame's audio position must
equal the total samples fed — an exact invariant (verified: 90,970 of 90,970 samples in the
live probe), not a statistical score. Offline capture of the delivered stream shows end-of-turn
audio/video drift within ±0.04 s. §7 discusses why instrument-level checks like these are
necessary but not sufficient — final acceptance of every synchronization change in this system
is a human watching the live avatar.

# 6. Evaluation

## 6.1 Setup and protocol

All measurements ran on the deployment machine itself: one NVIDIA RTX 5060 Ti (16 GB,
Blackwell), Windows 11 with the TTS server in WSL2. Configuration: CosyVoice2-0.5B on vLLM,
MuseTalk with TensorRT at 12 fps / 512 px output, steady (video-master) sync with a 14-frame
lead cushion, `llama-4-scout` pinned to one provider, first-clause splitting on for both
languages, filler words off.

Protocol: 10 runs per language (2026-07-16), each on a **fresh session** — the headless probe
reconnects per run, and the single-client renderer drops the previous session. Fresh sessions
are deliberate: we have observed long multi-turn sessions degrade turn-start latency (§7), so
a long-session campaign would conflate that open issue with the architecture's baseline.
Medians are the headline; p95 keeps the honest tail (GPU-contention variance and provider
congestion are both real and both appear in it). The question wavs are fixed recordings — a
zh "what is AI, explain in detail" prompt and an en weather-style prompt — so runs are
comparable within a language; the probe caveat for the LLM row is discussed in §4.

## 6.2 Latency results

**Table 3. TTFO and per-stage medians, 10 fresh sessions per language.**

| | zh | en |
|---|---|---|
| TTFO median | **2.92 s** | **2.20 s** |
| TTFO p95 | 4.94 s | 2.83 s |
| TTFO min / max | 2.14 / 4.94 s | 1.86 / 2.83 s |
| — TTS first chunk (median) | 0.89 s | 0.89 s |
| — sync lead-hold (median) | 0.64 s | 0.60 s |
| — transport + network (median) | 0.16 s | 0.17 s |
| — jitter + playout (estimate) | 0.15 s | 0.15 s |
| end-to-end to the ear (median) | 3.31 s | 2.75 s |

Both languages meet the 3 s TTFO target at the median. The zh tail is wider: its slow runs
attribute entirely to the sync lead-hold stretching under shared-GPU render variance (0.6 s
typical, up to 2.4 s on the worst run) — the TTS first chunk stayed within 0.66–1.04 s across
all 20 runs. This localizes future work precisely: the tail is a rendering-contention problem,
not a synthesis or network one.

## 6.3 Synchronization results

Across the 20 runs the probe received a median 12.1 fps (the configured rate is 12) with a
worst frame gap of 441 ms — below the 500 ms freeze threshold, and absorbed by the lead
cushion rather than audible as a voice pause. End-of-turn audio/video drift on offline
captures of the delivered stream is within ±0.04 s (development-time measurement, §5.4), and
the per-turn audio-position invariant (§5.4) held exactly on every probed turn. Sync
acceptance on live viewing is a human judgment and is discussed as a limitation in §7.

## 6.4 Resource footprint

**Table 4. Resident GPU memory, live stack (Windows per-process performance counters).**

| Process | Role | Dedicated GPU memory |
|---|---|---|
| WSL2 VM | CosyVoice2-0.5B on vLLM (KV pool at 7% of GPU) | 2,312 MiB |
| Renderer | MuseTalk + TensorRT (PyTorch weights freed) | 3,286 MiB |
| Pipeline | Pipecat + Silero VAD (CPU inference) | < 100 MiB |
| **Project total** | | **≈ 5.6 GB** |

The project's ~5.6 GB on a 16 GB card leaves headroom for a larger renderer or a local LLM;
conversely, with the documented settings (a slightly larger KV fraction on a smaller card),
the stack fits an 8 GB consumer GPU. The individual ablations behind this profile — vLLM KV
sizing, freed PyTorch weights, TensorRT — are the levers of Table 2 and §3.4.

# 7. Discussion and Lessons Learned

Three lessons from this system's measurement history generalize beyond it.

**Instrumented metrics and human perception diverge — in both directions.** Our automated
probe repeatedly passed configurations a human viewer rejected: a lead cushion of 8 frames
measured a clean, faster start on every probe metric, yet every value below 14 produced
visible delay or freezes live, and was abandoned. The converse also happened: a measured
difference in the synthesized Chinese waveform under CUDA-graph decoding led us to *predict* a
lip-sync degradation the viewer never saw, and the "fix" (disabling graphs) cost real latency
until the prediction was re-tested live and reversed. The rule we now operate by: a measurable
delta is neither necessary nor sufficient for a perceptual defect; instruments gate
regressions, but a human watching the live system is the arbiter in both directions.

**A verification's reference must not share the suspect input.** A long-lived lip-sync bug —
one corrupted byte offsetting every subsequent 16-bit audio sample sent to the renderer —
survived three debugging sessions because our checks compared the live render against an
offline render *fed the same corrupted stream*, or against audio captured downstream of the
repair point. Such tests are deterministic-consistency checks that cannot fail, mistaken for
end-to-end validation. Any audit of a processing chain must source its reference upstream of
every suspect component.

**Restore invariants at the producer, not the consumers.** That same byte-alignment invariant
(PCM frames must contain whole samples) was first patched at two consumer sites; both patches
were correct and the bug still recurred elsewhere. The durable fix was four lines at the single
point where network chunks become frames — carrying the dangling byte across reads — after
which every consumer patch was deleted. In a streaming pipeline, an invariant enforced where
data is created is one fix; enforced where data is used, it is one fix per consumer, forever.

**Limitations.** The renderer serves a single client; multi-viewer operation would need a
frame fan-out layer. MuseTalk generates the animated mouth region at 256×256 regardless of
output resolution — a model bound, not a transport one. Perceptual quality judgments in this
paper (sync acceptance, naturalness) come from a single expert viewer, not a formal MOS panel;
a small user study is the natural next step. Thai is supported through a separate TTS engine
(CosyVoice does not speak Thai) with known end-of-utterance truncation issues. Finally, on
long multi-turn sessions we have observed the turn-start latency degrade by ~1 s and not
recover; the cause is under investigation, which is why the evaluation protocol uses fresh
sessions.

# 8. Conclusion

VisualLLm demonstrates that a photoreal, multilingual, speech-to-speech talking-head agent is
achievable on one consumer GPU with open models and careful system architecture: stream every
stage, measure to the ear, and make audio–visual synchronization a declared per-frame contract
instead of a heuristic. The measured result — ~2.2–2.9 s median to the first synchronized
audio-and-video of an answer, in ~5.6 GB of VRAM — was reached not by a new model but by an
accumulation of individually measured architectural decisions, each reported here with its
cost or win. We believe the sync design (§5) and the measurement discipline (§4, §7) transfer
directly to other real-time multimodal systems; the natural next steps are a formal perceptual
study, multi-client serving, and closing the long-session degradation issue.

# References

<!-- generated from references.bib at conversion time -->
