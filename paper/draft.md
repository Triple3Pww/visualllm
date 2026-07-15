---
title: "VisualLLm: A Fully Local Streaming Architecture for Real-Time Multilingual Talking-Head Agents on a Single Consumer GPU"
author: "[AUTHORS — user to supply names, order, affiliations]"
bibliography: references.bib
---

# Abstract

<!-- Task 12 -->

# 1. Introduction

<!-- Task 12 -->

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
after which the originals are dead weight. The full stack holds ~7.8 GB, leaving half the card
free (Table 4); load order matters (TTS server before renderer) and is enforced by the
launcher.

## 3.5 Client

The browser client is a plain static page speaking standard WebRTC signaling; no build step.
An optional *split mode* streams only a fixed 256×256 mouth crop over the video track and
composites it client-side over a losslessly transmitted background still — concentrating the
video codec's bitrate budget on the only region that changes. A typed-input path and a
transcript view ride the same connection as side endpoints, leaving the pipeline structure
untouched.

# 4. Latency Engineering

<!-- Task 9 -->

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
[@tensorrt] cut the per-segment render cost from ~389 ms to ~255 ms (8-frame segments), the
difference between drifting seconds behind on long turns and holding ≥12 fps under load. Two
non-obvious details mattered: cuDNN autotuning (`cudnn.benchmark`) had to be disabled because
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

<!-- Task 11 -->

# 7. Discussion and Lessons Learned

<!-- Task 12 -->

# 8. Conclusion

<!-- Task 12 -->

# References

<!-- generated from references.bib at conversion time -->
