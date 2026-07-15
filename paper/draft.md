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

<!-- Task 10 -->

# 6. Evaluation

<!-- Task 11 -->

# 7. Discussion and Lessons Learned

<!-- Task 12 -->

# 8. Conclusion

<!-- Task 12 -->

# References

<!-- generated from references.bib at conversion time -->
