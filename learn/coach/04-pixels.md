# Module 4 — Pixels, GPUs, and the sync contract

## Who I am and what I'm doing

I'm Chanachon, a 3rd-year ICT student at Mahidol, on a research internship. I built a real-time system
where you speak into a browser and a photoreal talking-head avatar answers you out loud: microphone →
detect speech → speech-to-text → LLM → text-to-speech → face-animation model → back to the browser. It
runs on **one 16 GB GPU**, and the text-to-speech and the avatar share that single card. I built it with
heavy AI help and I'm learning the fundamentals underneath, one layer per week. So far: audio is int16 and
corruption is loud (M1); queues, backpressure, streaming, paced release (M2); autoregressive models,
prefill cost, sampling and repetition loops (M3).

**This is the last module — the avatar, the GPU they fight over, and how I keep the lips matching the
voice.** It ties the previous three together. Teach me the content, then quiz me. I have a small Python
exercise after, so explain ideas, don't write code for me.

**Out of scope — don't teach me:** writing GPU kernels, GPU microarchitecture, video codec internals, or
WebRTC negotiation. I need contention, compilation, and the sync contract.

---

## What I need to learn

### What a GPU is (minimally)
Thousands of small cores doing the same operation on lots of data at once. A **kernel** is a function you
launch onto it. Data must be copied into **VRAM** — the GPU's own memory — before it can be used. VRAM is
small and fixed; my card has 16 GB, and that's the whole budget.

### Contention — the fact behind most of my bugs
My text-to-speech and my avatar are **two separate processes on one card**. They fight over:
- **VRAM** — if either takes too much, the other fails to start. (In my system I *must* launch the TTS
  before the avatar, or it crashes with "no memory.")
- **Compute** — the GPU time-slices between them, so **both get slower**.

Consequence: **the avatar's real frame rate is a variable, not a constant** — it depends on what the TTS
is doing at that instant. Benchmarking the avatar alone tells me almost nothing about production, because
in production it shares the card with a TTS that's busiest at exactly the same moments (same turn).

### Frames and the frame budget
Video is still images at a fixed rate. At **14 fps** I have 1/14 = **71 ms** to produce each frame. Miss
the budget and I'm behind — and I don't get the time back.

### What the avatar model does per frame
1. Take a window of the **audio**.
2. Run a speech model over it to get features for *what sound is being made now*.
3. Feed those + the masked mouth region into a network that generates the mouth.
4. Decode to pixels, composite onto the face.

The key point: **the mouth is generated from the audio waveform.** So if the waveform is garbage, the
mouth is garbage — which is exactly why Module 1's one dropped byte destroyed the lip-sync (noise in →
generic flapping out) while the voice, a different copy, sounded perfect. This module closes that loop.

### TensorRT
An ahead-of-time **compiler** for neural networks: it fuses operations, picks the best kernels for my
specific GPU, uses lower precision (fp16). Compile once into an "engine" file; it runs much faster.
Costs: the engine is tied to that exact GPU and to fixed input shapes. In my project it cut render time
from ~389 ms to ~255 ms per segment.

### CUDA graphs
Normally the CPU launches every kernel one at a time, and that overhead adds up. A **CUDA graph** captures
a whole sequence of launches once and replays it as one unit — faster. But capture fixes the execution and
can subtly change numerical behaviour. (This is what perturbed my TTS sampling in Module 3's second bug.)

### The heart of it — the A/V sync contract
Audio plays at exactly one second per second; physics won't negotiate. Video comes from a renderer whose
speed is a variable — target 14 fps, but under GPU contention I actually get 10. So I produce 10 frames
for every 14 I need: **4 frames per second in debt, never repaid.**

How far behind are the lips after a long reply?
```
drift = turn_length × (1 − real_fps / target_fps)
```
In my exercise: **0.87 s at a 2 s turn, 3.27 s at 8 s, 8.07 s at 20 s** — dead linear. **This is why the
bug hid: short replies looked completely fine.**

There are exactly two ways to handle it:
- **Audio-master (`live`)**: play the voice immediately; show video whenever it arrives. Never pauses, but
  the lips slide progressively behind.
- **Video-master (`steady`)**: pin audio frame N to video frame N and release them together. Drift becomes
  *structurally impossible*, but the voice must wait for the renderer, so a stall becomes a silence.

Conventional media players (VLC, browsers) use **audio-master** — they drop video frames to keep audio
smooth, because people notice an audio glitch more than a dropped frame. **My system does the opposite**,
because a talking face whose mouth doesn't match is uncanny-valley wrong in a way a dropped frame isn't. A
pause is survivable; sliding lips aren't. The real lesson: **sync isn't a bug to fix, it's a contract you
choose** — both options are correct, they differ in who pays (the eyes or the ears).

### The twist — video-master alone isn't enough
If the renderer is *permanently* below target, video-master doesn't pause occasionally — it stretches the
*entire* voice. In my exercise, **every one of 111 audio gaps exceeds the budget** — it looks broken but
isn't; that's just what a permanently-too-slow renderer does. Run it with TensorRT (render now above
target): **0 of 111 gaps.** The conclusion I want to internalise: **TensorRT didn't fix the sync logic —
the logic was correct all along. It bought back the headroom that made the correct logic free.** Performance
work is often about buying enough slack that your correct design costs nothing.

---

## The bugs this explains ("P16" and "P1" in my project)

**P16** — the lips drifted further behind the longer the reply got, because the render rate was below target
and the deficit accumulated. Fixed with TensorRT — not by changing the sync, but by buying the headroom it
needed.

**P1** — a single flag, `cudnn.benchmark = True`, told the library to re-tune its algorithm on every new
input shape. The first segment of every turn has a different shape than mid-turn ones, so it re-tuned at the
*start of every turn* — a ~16-second GPU spike, every turn. Setting it to `False` removed it with no cost.
This is the same family as Module 3's bug: an optimisation that assumed a stable workload, in a system whose
workload is stable in the middle and never at the start. I've now seen this shape three times in three
layers (sampling, CUDA graphs, cudnn) — it's the most common way fast systems break.

---

## Questions for Gemini to ask me

1. My avatar benchmarks at 20 fps when run alone. What does that predict about production, and why?
2. What two resources do my TTS and avatar fight over, and what different failure does each cause?
3. Why does A/V drift grow with reply length instead of staying a constant offset?
4. Give the drift formula.
5. Audio-master vs video-master — what does each one sacrifice?
6. VLC uses audio-master. Why is my system right to do the opposite?
7. In video-master mode drift is zero — so where did the lost time go?
8. What did TensorRT actually fix? (Careful — it's not the sync logic.)
9. A benchmark says faster, a human says worse. Who wins, and what's the name of that failure?
10. **The closer:** the lip-sync model generates the mouth from the audio waveform. So what happens if one
    byte was dropped upstream? (I should be able to reconstruct Module 1's whole bug from scratch.)

Question 10 is the whole course. If I can answer it cold, I'm done. After this, I go back to my interactive
course for the capstone: trace one real turn end-to-end, then explain a bug I've never read before I read
the write-up.
