# Spec — A fundamentals study plan for VisualLLm

_2026-07-11. Author: Claude, with Chanachon. Status: design approved, ready to plan._

## Problem

Chanachon owns VisualLLm — 80 commits in the last two months, ~4,400 LOC of core
source, 44 numbered problem write-ups — but the system was built *with* an AI, and the
four layers it stands on (audio, async streaming, model inference, GPU systems) are
black boxes to him. Reading `STATUS.md` has not closed the gap; he asked for a study
plan that starts from fundamentals.

Self-assessed baseline: **zero comfort in all four foundations** (Python async, audio/PCM,
deep-learning inference, GPU systems). Budget: **~4 weeks, a few hours per week.**

## Goal

Transferable fundamentals, with VisualLLm as the running worked example. Success = he can
trace one turn end-to-end, name the layer/file/cost/knob at every hop, and explain any
P-number cold. The vocabulary must carry to his next project.

## Non-goals

Explicitly **not** taught: backprop and model training, audio codec internals, CUDA kernel
authoring, WebRTC RFC-level detail. This teaches the *slice* of each layer that VisualLLm
actually stands on, not a CS degree. Four weeks at a few hours a week cannot do more, and
pretending otherwise would produce a plan he abandons in week two.

## Shape: turn-trace spine, bug as payoff

Four modules follow **one conversation turn** through the system. Each module teaches its
foundation at the moment the turn crosses into that layer, then cashes the theory out on a
real failure from `docs/PROBLEMS-AND-FIXES.md`. Rejected alternatives: strict bottom-up
(three weeks of theory before he touches his own system — motivation dies), and pure
bug-driven (memorable but leaves gaps in the systematic map).

## The learning mechanism: predict, then run

The toy programs are **not** the deliverable. The **wrong prediction** is the deliverable.

Chanachon raised the right objection: if an AI can write the code, why write it? The answer
is that authoring the lines was never where the learning lived — *being corrected by reality*
was. Reading a concept produces the feeling of understanding, and that feeling is tested by
nothing. So every toy is gated behind a written prediction, recorded in `study/PREDICTIONS.md`
**before** anything executes. Where the prediction misses, that gap is the module. Where it
holds, move on fast.

Consequence for authorship:

| Toy | Who types it | Why |
|-----|--------------|-----|
| `study/m1_vad.py` | **Chanachon** | The lesson lives in the act of construction: he must write the byte-slicing loop himself and get noise out of it. Being told about P40 does not work — three sessions of tests that *could not fail* proved that. |
| `study/m2_pipeline.py` | Claude | He predicts the latency/queue behavior, then compares. Falsification survives without authorship. |
| `study/m3_sampler.py` | Claude | Same — predict the repetition loop, then watch it happen. |
| `study/m4_sync.py` | **Chanachon** | He must be the one who writes the release clock and watches video slide off audio. This is the system's central design decision. |

All toys are **stdlib-only** (`wave`, `asyncio`, `math`), run on system Python 3.11, need no
GPU and no conda env — a module must never die on setup. They point at real audio already in
`output/`.

## Modules

### Module 1 — Sound is just numbers

- **Fundamentals:** sampling; 16 kHz mono int16 PCM; what a chunk is; bytes↔seconds math;
  RMS/energy; why every sample is exactly two bytes and what one dropped byte does.
- **Toy (he types):** a from-scratch VAD. Read a WAV with stdlib `wave` (note: `q_ai.wav` is
  24 kHz, not 16 kHz — he must read the header, not assume), cut into 20 ms frames, threshold
  on RMS with a hangover, print speech start/stop. Then drop one byte and hear it become noise.
- **Real code:** `local_services/musetalk_video.py` (`_align_even`, `self._srv_carry`),
  `pipeline/stages/vad.py`.
- **Payoff:** **P40** — the lip-sync model was fed noise via odd-byte misalignment; the voice
  still sounded perfect, which is why it hid for three sessions. Also **P3** (the steady-mode
  screech: a discarded odd partial buffer).

### Module 2 — Streams, queues, and time

- **Fundamentals:** the event loop; coroutines; producer/consumer queues; backpressure; why
  streaming means the first token ships before the last one exists; paced release vs free-run.
- **Toy (Claude writes, he predicts):** a four-stage asyncio pipeline with bounded queues.
  Unbounded → latency blows up; bounded → backpressure; then a release clock emitting item N at
  `t0 + N/fps`. That clock *is* steady sync in miniature.
- **Real code:** `pipeline/main.py`, `pipeline/stages/*.py`, `local_services/first_piece_aggregator.py`,
  `pipeline/metrics.py`.
- **Payoff:** **P35** (our own `[TTFO]` metric measured the wrong end of the pipe and undercounted
  the real latency-to-the-ear by ~1.26 s) and the first-clause split that cut TTFO ~4.6→3.2 s (P23).

### Module 3 — Models that generate one step at a time

- **Fundamentals:** inference vs training; prefill vs decode, and why first-token latency scales
  with *input* length; sampling (top-p, repetition penalties); what a vocoder/flow decoder does;
  what vLLM buys; TTFB vs throughput.
- **Toy (Claude writes, he predicts):** a char-level Markov "model" that streams tokens. Show
  prefill cost growing with prompt length; greedy vs top-p; induce a repetition loop, then fix it
  with a repetition-aware sampler — P18 in miniature, with no torch.
- **Real code:** `local_services/cosyvoice_tts.py`, the RAS logits processor in the cosyvoice repo,
  `pipeline/stages/llm.py`.
- **Payoff:** **P18** (Chinese looped on silence because running the LLM on vLLM silently dropped
  CosyVoice's repetition-aware sampling), **P21** (the cloud LLM hop was the dominant TTFO cost and
  all its variance), **P43** (Traditional-Chinese garble → OpenCC t2s).

### Module 4 — Pixels, GPUs, and the sync contract

- **Fundamentals:** VRAM as the scarce resource; two processes contending for one card; what
  MuseTalk does per frame (crop → Whisper features → UNet → VAE → composite); what TensorRT
  compiles; what a CUDA graph captures and why capture can shift numerics.
- **Toy (he types):** a fake renderer slower than realtime, released two ways — audio-master
  (video free-runs and drifts) and video-master (audio paced to real rendered frames). He builds
  the system's central design decision with his own hands.
- **Real code:** `local_services/musetalk_server/app.py` (render loop, `video_clock`, `out_q`),
  `local_services/musetalk_video.py` (steady vs live), and the `video_out_is_live` coupling in
  `pipeline/main.py`.
- **Payoff:** **P16** (progressive drift → TensorRT is now baseline), **P1** (`cudnn.benchmark`
  causing a ~16 s spike on every turn's first segment), and **P33** — the stopwatch said *faster*,
  the eye said *worse*, and the eye won.

## Capstone (~2 hrs)

Run `python -m scripts.measure --offline-capture` on a real turn and write the latency waterfall
himself: for every hop, name the layer, the file, the milliseconds, and the one knob that moves it.
Then Claude picks a P-number at random and he explains it cold.

## Deliverables

1. `docs/STUDY-PLAN.md` — the curriculum. Per module: concept → predict → build/run → compare →
   real-code trace → the P-number it explains → five self-check questions he can grade alone.
2. `study/` — `m1_vad.py` (his), `m2_pipeline.py` (Claude's), `m3_sampler.py` (Claude's),
   `m4_sync.py` (his), plus `PREDICTIONS.md` (written before each run) and a `README.md`.

Both are git-tracked in this repo, next to the code they teach.

## Success criteria

- He traces one turn end-to-end unaided, naming layer/file/cost/knob at every hop.
- He explains a randomly-chosen P-number cold.
- He can answer the "would I have caught it in review?" test — spotting an unbounded queue or an
  odd-byte drop — not merely define the terms.
