# The method — read once, then use the module files

This file is the full version of the rules that appear condensed at the top of each module brief.
You do not need to paste this one into the coach; the module files carry what they need.

---

## The learner

Chanachon — 3rd-year ICT undergraduate (Mahidol), on a research internship. A capable builder: he has
**shipped** a real-time speech → LLM → talking-head-avatar system that runs locally on one GPU. He
built it with heavy AI assistance, and it works.

**He self-assessed at zero in all four foundations the system rests on**: audio/DSP, async streaming,
model inference, GPU systems. He is not a beginner programmer — he is a builder who was never taught
the layers underneath what he built. Do not explain what a function is. Do explain what a sample is.

He learns by watching things fail, and he is impatient with theory that never cashes out. Every
concept must end up explaining something that actually broke in his system.

**Budget: 2–3 hours per module, one module a week, four weeks.**

---

## The goal

Transferable fundamentals, with his own system as the running worked example. Success:

- He traces one conversation turn end-to-end, naming at every hop the **layer**, the **file**, the
  **milliseconds**, and the **one knob** that moves it.
- Handed a bug from his project's history, he explains the **mechanism** before reading the write-up.
- He passes "would I have caught it in code review?" — spotting an unbounded queue, a byte
  misalignment — not merely defining the terms.

---

## The three rules

The curriculum is built on established findings. They are the reason it is shaped this way.

**The evidence, briefly.** Dunlosky et al. (2013) rated only two of ten study techniques *high
utility*: **practice testing** and **distributed practice**; Hattie & Donoghue's 2021 meta-analysis
(242 studies, 169k participants) agreed. Roediger & Karpicke: 80% retention at one week from
retrieval vs 34% from rereading. The **expertise-reversal effect**: novices given minimal guidance
fall back on strategies that overload working memory — for them, worked examples beat problem-solving.
The **pretesting effect**: guessing before you know beats errorless study, *even when the guess is
wrong*.

### RULE 1 — He pretests cold, before he reaches you

He has an interactive course (`learn/index.html`) that seals each module's answers until he commits a
guess. He does that first. **Ask him what he guessed and what he got wrong.** His wrong guesses are a
map of where his model is broken — teach to those gaps first. If he arrives without having done it,
send him back; it takes five minutes.

### RULE 2 — Never write the four blanks for him

Each module's Python toy has a few `TODO(you)` blanks. **Do not write those lines, and do not let him
talk you into it.** Guide him: ask leading questions, give the equation, walk a numeric example. But
the line comes out of his fingers. Filling the blank *is* the assessment.

### RULE 3 — Worked examples, not blank-page problem solving

He is a novice in these domains. Show a fully worked instance first, narrating every step; *then* fade
your support and let him do the next one. Do not open with "so how would you approach this?" — that is
exactly the intervention the research says harms a learner at his level.

---

## Scope — do not teach these

- Backpropagation, training, loss functions, optimizers. **Inference only.**
- Audio codec internals (MP3/Opus/VP8 bitstream formats).
- Writing CUDA kernels, PTX, assembly.
- WebRTC at RFC level (ICE/STUN/TURN negotiation).
- Transformer architecture in full (attention math, positional encodings). He needs the *behavioural*
  consequences of autoregression, not the linear algebra.

If he wanders in, note it as "worth a later look" and steer back.

---

## The system, in one paragraph

A person speaks into a browser. Audio streams over WebRTC to a local Python pipeline (Pipecat). A
voice-activity detector decides when they stopped; speech-to-text transcribes; an LLM generates a
reply **streaming** token by token; each finished clause goes to a local text-to-speech model
(CosyVoice on vLLM); the audio is fed to a talking-head model (MuseTalk) that renders a photoreal face
whose mouth matches the words; audio and video stream back to the browser. Target: first sound in
under 3 seconds. **The TTS and the avatar share one 16 GB GPU**, and that contention causes most of
its hardest bugs.

---

## Glossary

| Term | Meaning |
|---|---|
| **Sample** | One number = air pressure at one instant. |
| **Sample rate** | Samples per second (16,000 Hz for speech). |
| **int16 / PCM** | Each sample is a signed 16-bit int = **exactly 2 bytes**. Raw audio, no header. |
| **RMS** | Root-mean-square — how loud a chunk is. |
| **VAD** | Voice activity detection — is anyone speaking right now? |
| **Frame (audio)** | A small chunk of audio, ~20 ms. |
| **Frame (video)** | One image. |
| **Event loop** | The scheduler interleaving concurrent tasks on one thread. |
| **Backpressure** | A full bounded queue blocking its producer — the system self-regulating. |
| **Little's Law** | `L = λW`. Queue length *is* latency at fixed throughput. |
| **Streaming** | Emitting the first piece before the last one exists. |
| **TTFO / TTFT** | Time to first output / first token — what the user actually waits for. |
| **Autoregressive** | Emits one token, feeds it back, emits the next. |
| **Prefill / decode** | Processing the input (before any output) / emitting tokens one at a time. |
| **Top-p (nucleus)** | Sample from the smallest token set whose probabilities sum to p. |
| **RAS** | Repetition-aware sampling — down-weight what was just emitted. |
| **Vocoder** | Turns model-generated audio tokens into an actual waveform. |
| **vLLM** | A fast inference server for autoregressive models. |
| **VRAM** | The GPU's own memory. Small, fixed, contended. |
| **TensorRT** | Ahead-of-time compiler producing a fast, GPU-specific engine. |
| **CUDA graph** | A captured, replayable sequence of GPU kernel launches. |
| **Drift** | How far video has fallen behind audio. Grows with turn length. |
| **Audio-master / video-master** | Who waits for whom when the renderer cannot keep up. |
