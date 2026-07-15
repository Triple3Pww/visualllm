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

<!-- Task 8 -->

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
