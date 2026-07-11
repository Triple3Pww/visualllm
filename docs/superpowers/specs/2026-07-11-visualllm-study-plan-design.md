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

## The learning mechanism (revised against the evidence, 2026-07-11)

Chanachon asked the right question — *if an AI can write the code, why write it?* — and asked
for the research rather than accepting an intuition. The research overturned two of this spec's
original choices. What follows is the evidence-grounded design; the rejected v1 is recorded at
the bottom so the reasoning isn't lost.

**Finding 1 — only two techniques are high-utility.** Dunlosky et al. (2013) rated just two of
ten study techniques as high-utility: **practice testing** and **distributed practice**. Hattie &
Donoghue's 2021 meta-analysis (242 studies, 169k participants) concurred. Roediger & Karpicke
(2006): after one week, retrieval-practice students retained **80%** vs **34%** for rereading.
*v1 contained neither as a structural element.* This was its largest defect.

**Finding 2 — expertise reversal.** For low-knowledge learners, studying **worked examples**
beats minimal-guidance problem solving; novices handed a blank file "resort to inefficient
problem-solving strategies that overwhelm working memory." Chanachon self-assessed at **zero in
all four domains**, so v1's "type `m1_vad.py` from scratch" was precisely the intervention the
evidence says harms a learner in his position.

**Finding 3 — but concept-only is worse.** That is the 34% number. Passive reading yields
fluency without capability, which is the failure mode he is *already in* (he has read `STATUS.md`
and it did not stick). "Just learn the concept" correctly diagnoses that authoring lines is not
the point, then lands on the wrong remedy.

**Finding 4 — the reconciliation.** *Faded worked examples / completion problems* (Renkl &
Atkinson): begin from a fully worked solution, progressively blank out steps, end unaided. Load
stays bounded and the fade **cures** expertise reversal by tracking growing skill. And, answering
his literal question, *Parsons problems* (CS-education): novices rearranging scrambled correct
code learn **the same amount** as those writing from scratch, but **faster, at lower cognitive
load, and without getting stuck** — the measured downside (resentment) appears only in
*experienced* students.

**Finding 5 — predict-then-run survives, and moves earlier.** The **pretesting effect**: guessing
before you know — *even when the guess is wrong* — beats errorless study, with or without
feedback, across retention intervals. So predictions are kept and promoted to a **cold pretest at
the start of each module, before the reading**.

### What this changes

The toys are **not** the deliverable and the **wrong prediction** still is. But nobody types a
toy from a blank file. Every toy ships as a **faded worked example**: complete code with
`# TODO(you):` blanks whose density **increases across the four weeks** (M1 mostly given → M4
mostly blank), so guidance recedes exactly as his expertise grows.

The key correction v1 got wrong: **the byte-drop demo never required him to have authored the
chunker.** Getting noise out of a one-byte mutation is a *falsification*, not an authoring task.
The full punch of P40 is preserved with the blank-file overload removed — v1 falsely paired them.

### Module structure (every module, in order)

1. **Cold pretest (~10 min)** — predict answers *before any reading*, into `learn/PREDICTIONS.md`.
   Wrong guesses are the point (Finding 5).
2. **Spaced interleaved retrieval (~10 min)** — cold questions drawn from *previous* modules, not
   this one (Finding 1). Module 1 skips this; it has no past.
3. **Concept** — the reading, now aimed at the gaps the pretest exposed.
4. **Faded worked example** — run the toy, fill its `TODO(you)` blanks, with self-explanation
   prompts ("why this line?") after each worked step (Renkl).
5. **Falsify** — mutate and run: drop the byte, unbound the queue, remove the sampler's penalty.
   Compare against the prediction. The gap is the module.
6. **Trace the real code** — the same idea, in the actual repo file.
7. **Bug payoff** — the P-number it explains, read only *after* he can already predict it.
8. **Five self-check questions** — which become week N+1's retrieval quiz. Nothing is one-and-done.

**Spacing beyond the four weeks:** two cold retrieval checkpoints at **week 6** and **week 10**
(~15 min each, capstone questions only). Distributed practice is what converts this from a
month of activity into durable memory; without it the forgetting curve eats the whole plan.

All toys are **stdlib-only** (`wave`, `asyncio`, `math`), run on system Python 3.11, need no
GPU and no conda env — a module must never die on setup. They point at real audio already in
`output/` (note `q_ai.wav` is 24 kHz, not 16 kHz: the toy must read the header, not assume).

## Modules

### Module 1 — Sound is just numbers

- **Fundamentals:** sampling; 16 kHz mono int16 PCM; what a chunk is; bytes↔seconds math;
  RMS/energy; why every sample is exactly two bytes and what one dropped byte does.
- **Toy (faded, ~20% blank):** a VAD supplied nearly complete. He fills the frame-slicing and
  RMS blanks, reads the WAV header rather than assuming 16 kHz, and prints speech start/stop.
- **Falsify:** drop **one byte** and re-run — speech becomes noise. This is an experiment, not
  an authoring task; it lands with full force regardless of who typed the chunker.
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
- **Toy (faded, ~70% blank — the heaviest fade, by design):** a fake renderer slower than
  realtime, released two ways — audio-master (video free-runs and drifts) and video-master (audio
  paced to real rendered frames). By week 4 the guidance has receded far enough that he writes the
  release clock itself: the system's central design decision, built with his own hands, but
  *earned* rather than dropped on him cold in week 1.
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

## Deliverables — one self-contained `learn/` folder

Everything for learning lives in **one folder at the repo root**, not scattered between `docs/`
and a separate toy directory. He opens `learn/` and works out of it; each module's guide sits
directly beside the toy it drives.

```
learn/
  README.md          <- START HERE: the 4-week schedule, how a module runs, the spacing checkpoints
  PREDICTIONS.md     <- his log. Cold pretest answers go here BEFORE any reading. Never edited after.
  m1-sound.md        <- Module 1 guide      m1_vad.py       <- faded toy (~20% blank)
  m2-streams.md      <- Module 2 guide      m2_pipeline.py  <- faded toy (~40% blank)
  m3-models.md       <- Module 3 guide      m3_sampler.py   <- faded toy (~55% blank)
  m4-pixels.md       <- Module 4 guide      m4_sync.py      <- faded toy (~70% blank)
  capstone.md        <- the end-to-end turn trace + the random P-number defense
  ANSWERS.md         <- self-check answer keys, so he can grade himself without Claude
```

Each module guide follows the eight-step structure above (pretest → spaced retrieval → concept →
faded example → falsify → real-code trace → bug payoff → five self-check questions). `README.md`
is a lean router in the house style: the schedule and the method, pointing at the module files —
not an encyclopedia.

Git-tracked in this repo, so the curriculum sits next to the code it teaches and travels with it.
`learn/` is deliberately a sibling of `pipeline/` and `local_services/`, not buried in `docs/` —
it is a place to *work*, not a document to read.

## Success criteria

- He traces one turn end-to-end unaided, naming layer/file/cost/knob at every hop.
- He explains a randomly-chosen P-number cold.
- He can answer the "would I have caught it in review?" test — spotting an unbounded queue or an
  odd-byte drop — not merely define the terms.
- **He still passes the week-10 cold retrieval check.** Anything that decays before then was
  activity, not learning.

## Rejected: v1's "type two toys from scratch"

The first draft of this spec had Chanachon hand-author `m1_vad.py` and `m4_sync.py` from a blank
file, on the reasoning that "the lesson lives in the act of construction." The evidence says that
is true for *experienced* learners and false for novices — for whom it induces working-memory
overload and open-ended time sinks (expertise reversal; the Parsons-problem efficiency results).
It also rested on a false pairing: the P40 byte-drop shock was assumed to require authorship, when
it only ever required *falsification*. Recorded here so the reasoning isn't rediscovered and
repeated. Faded worked examples deliver the same construction experience with bounded load, and
by week 4 the fade reaches near-blank anyway — so nothing of value was lost, only the overload.

## Evidence

- Dunlosky et al. (2013), *Improving Students' Learning With Effective Learning Techniques* —
  practice testing + distributed practice are the only two high-utility techniques of ten.
  <https://www.researchgate.net/publication/258180568>
- Hattie & Donoghue (2021 meta-analysis, 242 studies / 169k participants) — "the most effective
  techniques are Distributed Practice and Practice Testing."
  <https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2021.581216/full>
- Roediger & Karpicke (2006) — 80% retention at one week via retrieval vs 34% via rereading.
- Kalyuga et al., *The Expertise Reversal Effect* — novices benefit from full guidance; the
  advantage reverses as knowledge grows. <https://www.uky.edu/~gmswan3/EDC608/Kalyuga2007_Article_ExpertiseReversalEffectAndItsI.pdf>
- Renkl & Atkinson, *From Studying Examples to Solving Problems: Fading Worked-Out Solution Steps
  Helps Learning*. <https://www.researchgate.net/publication/2398854>
- Ericson et al. (2021), *Problem-Solving Efficiency and Cognitive Load for Adaptive Parsons
  Problems vs. Writing the Equivalent Code* — equal learning, higher efficiency, lower load.
  <https://dl.acm.org/doi/fullHtml/10.1145/3411764.3445292>
- Richland, Kornell & Kao, *The Pretesting Effect: Do Unsuccessful Retrieval Attempts Enhance
  Learning?* — yes; errorful generation beats errorless study.
  <https://learninglab.uchicago.edu/Pre-Testing_files/RichlandKornellKao.pdf>
