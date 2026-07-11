# Coach brief — teaching the four foundations behind VisualLLm

**Hand this whole file to the learning coach (Gemini).** It is self-contained: the coach does not
have the codebase, so everything needed is here.

---

## 1. Who you are teaching

Chanachon — a 3rd-year ICT undergraduate (Mahidol), currently on a research internship. He is a
capable builder: he has *shipped* a real-time speech→LLM→talking-head avatar system that runs
locally on one GPU. He built it with heavy AI assistance, and it works.

**But he has self-assessed at zero in all four of the foundations the system rests on**: audio/DSP,
async streaming, model inference, and GPU systems. He is not a beginner programmer — he is a
builder who has never been taught the layers underneath what he built. Teach accordingly: do not
explain what a function is; *do* explain what a sample is.

He learns by seeing things fail. He is impatient with theory that never cashes out. Every concept
you teach should end up explaining something that actually broke in his system.

**Time budget: about 2–3 hours per module, one module per week, four weeks.**

---

## 2. What he is trying to achieve

Transferable fundamentals, with his own system as the running worked example. Success means:

- He can trace one conversation turn end-to-end and name, at every hop, the **layer**, the **file**,
  the **cost in milliseconds**, and the **one knob** that moves it.
- He can be handed a bug report from his own project's history and explain the **mechanism** before
  reading the write-up.
- He passes the "would I have caught it in code review?" test — spotting an unbounded queue or a
  byte-misalignment — not merely defining the terms.

---

## 3. The method — and three rules you must not break

The curriculum is built on established learning-science findings. Please respect them; they are the
reason it is shaped this way.

- **Retrieval practice and spaced practice** are the only two techniques rated *high utility* in
  Dunlosky et al. (2013), confirmed by Hattie & Donoghue's 2021 meta-analysis. Roediger & Karpicke:
  80% retention at one week from retrieval vs 34% from rereading.
- **The expertise-reversal effect**: novices given minimal guidance resort to inefficient strategies
  that overload working memory. For a low-knowledge learner, *worked examples beat problem-solving.*
- **The pretesting effect**: guessing before you know — *even when the guess is wrong* — beats
  errorless study.

### RULE 1 — He takes a cold pretest BEFORE he comes to you.

He has an interactive course (`learn/index.html`) that seals each module's answers until he commits
a guess. He does that first, *then* brings the module to you. **Ask him what he guessed and what the
page told him.** His wrong guesses are the single most valuable input you have: they are a map of
exactly where his model is broken. Teach to those gaps first.

If he shows up without having done it, send him back. It takes five minutes.

### RULE 2 — Never hand him the code for the four blanks.

Each module has a Python "toy" with a small number of `TODO(you)` blanks he must fill. **Do not write
those lines for him, and do not let him talk you into it.** Guide him to derive them — ask leading
questions, give him the equation, walk a numeric example — but the line must come out of his fingers.
Filling the blank *is* the assessment. (The blanks are listed per-module below so you know what they
are. They are deliberately small: 1–3 lines each.)

### RULE 3 — Worked examples, not blank-page problem solving.

He is a novice in these domains. Show him a fully worked instance first, narrate every step, *then*
fade your support and let him do the next one. Do not open with "so how would you approach this?" —
that is the intervention the research says harms learners at his level.

---

## 4. Scope — what NOT to teach

Four weeks at a few hours a week. Deliberately out of scope; please skip, even if he asks:

- Backpropagation, training, loss functions, optimizers. **Inference only.**
- Audio codec internals (MP3/Opus/VP8 bitstream formats).
- Writing CUDA kernels; PTX; assembly.
- WebRTC at RFC level (ICE/STUN/TURN negotiation detail).
- Transformer architecture in full (attention math, positional encodings). He needs the *behavioural*
  consequences of autoregression, not the matrix algebra.

If he wanders into these, note it as "worth a later look" and steer back.

---

## 5. The system he built, in one paragraph (context for you)

A person speaks into a browser. Audio streams over WebRTC to a local Python pipeline (Pipecat).
A voice-activity detector decides when they stopped talking; speech-to-text transcribes it; an LLM
generates a reply, **streaming** token by token; each finished clause is sent to a local text-to-speech
model (CosyVoice, running on vLLM); the resulting audio is fed to a talking-head model (MuseTalk) that
renders a photoreal face whose mouth matches the words; audio and video stream back to the browser.
Target: first sound out of the speakers in under 3 seconds. The TTS and the avatar **share one 16 GB
GPU**, and that contention is the source of most of its hardest bugs.

---

# MODULE 1 — Sound is just numbers

**Toy:** `learn/m1_vad.py` · **2 blanks** · **The bug it explains: "P40"**

## Objective

By the end he understands that audio is an array of integers, can convert freely between bytes and
seconds, and understands *viscerally* why a one-byte misalignment destroys a signal while leaving it
sounding plausible in every measurement he'd naively take.

## Teach, in this order

**1. What a sample actually is.** Sound is a pressure wave. A microphone converts pressure into
voltage, continuously. A computer cannot store "continuous", so it *samples*: it measures the voltage
at fixed intervals and writes down a number. That number is a **sample**. That is the entire idea.

**2. Sample rate.** How many measurements per second. 16,000 Hz means 16,000 numbers per second of
audio. Motivate it: the Nyquist–Shannon theorem says you can faithfully represent frequencies up to
*half* your sample rate, so 16 kHz captures up to 8 kHz — which covers the frequencies that make
speech intelligible. That is why speech systems standardise on 16 kHz while music uses 44.1 kHz.

**3. Bit depth and why every sample is exactly two bytes.** Each sample is an `int16`: a signed
16-bit integer, range −32,768 to +32,767. Sixteen bits = **two bytes**. Have him say this out loud.
It is the load-bearing fact of the entire module.

**4. Channels.** Mono = one stream of samples. Stereo = two, interleaved (L,R,L,R…). His system is
mono throughout. Mention interleaving only so he knows why channel count enters the byte equation.

**5. THE EQUATION.** Derive it with him, do not just state it:

```
bytes = seconds × sample_rate × channels × bytes_per_sample
```

For his system (mono, int16): `bytes = seconds × rate × 2`.

Now make him run it **in both directions**, several times, until it is automatic:
- Forward: "3 seconds at 16 kHz mono int16 is how many bytes?" → 3 × 16000 × 2 = **96,000**.
- Backward: "I am at byte offset 48,000 in a 16 kHz mono int16 stream. What time is that?"
  → 48000 / (16000 × 2) = **1.5 seconds**.

This backward direction — byte offset to timestamp — is how every audio system on earth knows
"where" it is. Drill it.

**6. PCM vs WAV.** PCM is the raw array of samples, nothing else. A WAV file is that array plus a
small header saying the rate, the channels, and the bit depth. **The header is the only reason
anyone knows how to interpret the bytes.** Which sets up the trap in the toy: his test file is
**24,000 Hz**, not the 16,000 his pipeline uses. If he assumes 16 kHz instead of reading the header,
every timestamp he computes is 1.5× wrong and nothing crashes.

**7. Endianness — briefly.** The two bytes of an `int16` have an order. x86 is little-endian (low
byte first). He needs this only to understand *why* a decoder must know where a sample **starts**.

**8. Framing.** Real-time audio is processed in small chunks called **frames**, typically 10–30 ms.
Why not sample by sample? Too much overhead. Why not one big block? You would not be real-time
anymore. 20 ms is the usual compromise. At 16 kHz mono int16, one 20 ms frame = 0.020 × 16000 × 2 =
**640 bytes**. Make him compute that himself.

**9. Energy / RMS.** How loud is a frame? Take the root-mean-square: square every sample, average,
square-root. It is just "typical magnitude", robust to the fact that a waveform swings positive and
negative (a plain average would be ≈ 0). Optional: mention dB as the log scale of this.

**10. Voice activity detection.** The simplest VAD: if a frame's RMS is above a threshold, call it
speech. Then explain **hangover** — why a naive threshold chops a sentence into fragments, because
natural speech contains short pauses between words. So you keep the "speech" state alive for N quiet
frames before declaring it over. His toy implements exactly this. (His real system uses **Silero**, a
small neural VAD, because energy thresholds fail on background noise — but the *shape* of the
algorithm is identical.)

**11. THE PUNCHLINE — byte misalignment.** Now the thing this module exists for.

If every sample is exactly two bytes, then the byte stream must be cut on **even** boundaries. Drop
or insert a **single byte**, and every subsequent `int16` gets assembled from the *second half of one
sample and the first half of the next*. Walk it byte by byte on paper with him — draw the boxes.

Ask him to predict what that sounds like. Most people say "a small glitch" or "some crackle."

The truth: it is **noise** — and, crucially, it is **louder** than the speech was. In his toy, one
dropped byte takes the mean RMS from **1,156 to 16,547** (14× louder), and the detected speech
structure collapses. Make sure he sees *why* it gets louder: pairing unrelated bytes produces
essentially random 16-bit values, which are on average far larger in magnitude than a real speech
waveform, and they flip sign every sample — high-frequency, high-energy hash.

**Sit on this.** It is the module. "Corruption is loud, not quiet" is the insight that makes the
real bug comprehensible.

## The blanks he must fill (DO NOT write these for him)

1. `frame_size_bytes(rate, sample_width, frame_ms)` — return the number of **bytes** in one frame.
   He has the equation from step 5. If he is stuck, ask "how many *samples* is 20 ms at this rate?"
   and then "and how many bytes is each sample?"
2. `rms(frame)` — square, mean, square-root, over the unpacked `int16` samples. If he is stuck, ask
   him what R, M, and S stand for and have him do it in that order, backwards.

## Misconceptions to hunt and kill

- *"A dropped byte is a minor glitch."* No — it destroys every subsequent sample. It is total.
- *"If it sounds fine, it is fine."* This is the killer. See the bug below.
- *"Loudness correlating with mouth movement proves the lip-sync works."* No: **noise has RMS too.**
  Corrupt audio is *louder* than clean speech, so a mouth flapping at garbage correlates beautifully
  with the loudness of that garbage. In his project this false test misled them **four separate times.**

## The bug this explains — "P40"

His TTS returns audio buffers that sometimes have an **odd** number of bytes. Two copies of that audio
are made: one is sent to the browser (what the human hears), one is sent to the lip-sync model.

The browser copy was protected against the odd byte. **The avatar copy was not** — it dropped the
stray byte. So the lip-sync model was being fed pure noise, for months. And because the model
generates mouth shapes by running speech recognition *on the waveform it is given*, it produced a
generic, wordless flapping that never closed for pauses.

**The voice always sounded perfect.** That is the trap. The bug lived entirely in the branch he could
not hear.

It survived **three separate debugging sessions** because every test written to catch it *could not
fail*: the offline "reference render" was fed audio captured from the browser — i.e. the *repaired*
copy — so it always looked fine, and proved only that the renderer was deterministic.

**Ask him to state the lesson himself.** The one you're fishing for is roughly: *a test whose
reference shares the suspect input cannot fail, and a passing test that cannot fail is worse than no
test, because it buys false confidence.*

## Exit criteria

- Converts bytes↔seconds in both directions, from memory, without hesitation.
- Explains why one dropped byte produces *loud* noise rather than silence or a click.
- Explains why the P40 tests could not fail, and can propose a test that *could* have.

---

# MODULE 2 — Streams, queues, and time

**Toy:** `learn/m2_pipeline.py` · **3 blanks** · **The bug it explains: "P35"**

## Objective

He understands the event loop, producer/consumer queues, and backpressure; he understands why
streaming decouples first-output latency from total length; and he learns that a metric that stops
before the user does is not a metric.

## Teach, in this order

**1. Concurrency is not parallelism.** Parallelism = doing two things at literally the same instant
(two cores). Concurrency = *making progress* on many things by interleaving them on one core. His
pipeline is concurrent, mostly single-threaded, and that is fine — because the stages spend most of
their time **waiting** (for the network, for the GPU), not computing.

**2. Blocking vs non-blocking.** A blocking call (`time.sleep`, a plain socket read) stops the whole
thread. If a real-time loop blocks, everything downstream starves. This is why an audio pipeline can
be destroyed by one careless synchronous call.

**3. The event loop, coroutines, `await`.** Teach it as a to-do list. The loop holds a list of tasks.
Each task runs until it hits an `await` on something not ready yet — at which point it *voluntarily
hands control back*, and the loop runs someone else. When the awaited thing is ready, the task is
resumed. That is 90% of `asyncio`.

Emphasise the word **voluntarily**: a task that never awaits (a tight CPU loop) starves everything.

**4. Producer / consumer queues.** Stages connect through queues. Stage A puts, stage B gets. The
queue decouples their speeds — that is the *point* of it, and also its danger.

**5. THE CORE LESSON — bounded vs unbounded, and backpressure.**

Set up the scenario and make him predict before you reveal: a producer that makes an item every
50 ms, feeding a consumer that takes 120 ms per item. The consumer is 2.4× too slow. The queue has
no size limit.

Nothing crashes. Nothing is dropped. **So what goes wrong?**

The answer he must reach on his own: **latency**. The queue absorbs the mismatch by growing, so by
the end you are delivering something made *seconds ago*. An unbounded queue does not fix a slow
consumer — **it hides it, and bills you in staleness.**

Now bound the queue to 2 items. `await queue.put()` cannot complete when the queue is full, so it
**blocks the producer** — forcing the whole pipeline down to the speed of its slowest stage. That
blocking is called **backpressure**, and it is a feature, not a bug.

His toy measures it: worst item age drops from **1.30s to 0.45s** — nearly 3× fresher — while total
wall time is **identical** (2.52s vs 2.56s). Make him confront that: throttling the producer cost
*nothing* in throughput and bought everything in latency. Almost everyone predicts this backwards.

**6. Little's Law — give him this, it's the fundamental underneath.**

```
L = λ × W        (items in system = arrival rate × time in system)
```

Rearranged: **W = L / λ.** Time-in-system is proportional to *queue length*. So a longer queue is
*mathematically* more latency at the same throughput. This is not a heuristic; it's an identity. It
is why "just make the buffer bigger" is so often exactly the wrong instinct.

**7. Throughput vs latency.** They are different, they trade off, and optimizing one can wreck the
other. Batching raises throughput and raises latency. This distinction returns hard in Module 3.

**8. Streaming.** The LLM does not produce a whole answer and hand it over; it emits tokens one at a
time. So the first *sentence* can be sent to the TTS before the rest of the answer exists, and the
first *audio chunk* can be sent to the avatar before the sentence is finished synthesising.

The consequence he must internalise: **time-to-first-output depends on the first piece, not on the
total length.** His system exploits this deliberately — it splits off a short opening clause and
speaks it early, which cut time-to-first-sound from ~4.6s to ~3.2s even though the full answer takes
exactly as long as before.

**9. Paced release vs free-run.** Two ways to emit: the instant it's ready (free-run), or on a fixed
clock — item N leaves at `t0 + N/fps` (paced). Flag this hard: **it is the central design decision of
Module 4.** His toy has him build the release clock.

**10. Where do you stop the stopwatch?** Set up the bug below by asking: "if you want to measure how
long the user waits to hear a reply, where does the timer stop?"

## The blanks he must fill (DO NOT write these for him)

1. Put `(i, timestamp)` on the queue using the **async** put (not `put_nowait`) — the `await` *is*
   the backpressure. If he uses `put_nowait`, the bounded case will silently behave like the
   unbounded one. Ask him why.
2. Compute the release `deadline` for item n: `start + n / FPS`.
3. Sleep only if early; never sleep a negative duration. (Ask what happens if you do.)

## Misconceptions to hunt

- *"A bigger buffer is safer."* Little's Law says a bigger buffer is more latency.
- *"Throttling the producer will make the system slower."* Measured: identical wall time, 3× fresher.
- *"Latency is just 1/throughput."* No. A system can have huge throughput and terrible latency.
- *"If the server says it started speaking, the user heard it."* No — see below.

## The bug this explains — "P35"

His system measured time-to-first-output as the gap between "user stopped speaking" and a
`BotStartedSpeaking` event — an event emitted **inside the server**.

But the user's ear is on the far side of a transport layer, a video encode, a network hop, and a
browser jitter buffer. All of that happens **after** the stopwatch stopped. It was worth **1.26
seconds**.

So the team spent weeks optimising hard against a number that did not describe the experience they
were shipping. Every value it reported was *true*. The thing it measured simply wasn't the thing that
mattered.

**The lesson, in his words:** *a metric that stops before the user does is not a metric, it is a
comfort.*

## Exit criteria

- Explains backpressure without using the word "wait" as the whole answer.
- States Little's Law and uses it to argue against a bigger buffer.
- Explains why TTFO is independent of reply length in a streaming system.
- Can identify, for any metric put in front of him, whether it stops where the user is.

---

# MODULE 3 — Models that generate one step at a time

**Toy:** `learn/m3_sampler.py` · **4 blanks** · **The bug it explains: "P18"**

## Objective

He understands autoregressive generation and the two consequences that dominate his system: **why
first-token latency scales with input length**, and **why a model can lock into a repetition loop** —
plus what sampling policy is and why it is a correctness feature, not a style knob.

> **This is the hardest module for someone with no deep-learning background. Budget more time here
> than the others, and go slower.** If it needs two weeks, give it two weeks.

## Teach, in this order

**1. What a model *is* at inference.** A function with learned parameters. Given a context, it
outputs a score for every possible next token. That's it. No learning happens at inference; the
weights are frozen. He does not need to know how they got that way.

**2. Tokens.** Text is chopped into tokens (roughly: common words and word-pieces). The model has a
fixed vocabulary of maybe 30k–100k of them. It emits **one at a time**.

**3. Autoregression — the whole ballgame.** The model predicts the next token given everything so
far. Then that token is **appended to the input**, and the model runs again. And again. Its own
output becomes its next input.

Write the loop on the board. Every strange behaviour in this module is a consequence of this loop.

**4. Prefill vs decode.** Before the model can emit token #1, it must process **every token of the
input**. That is **prefill**. Emitting tokens one by one afterwards is **decode**.

The consequence he must own: **time-to-first-token scales with the length of the INPUT, not the
output.** A long prompt costs a late first token even if the answer is one word.

His toy demonstrates it directly: prompts of 20 / 200 / 1500 characters produce first-token times of
**0.016s / 0.109s / 0.813s** — before a single character of output exists.

*(If he's ready for it: prefill is parallel over the input — all tokens at once, compute-bound — while
decode is inherently sequential, one token at a time, and memory-bandwidth-bound. That's why they have
such different performance characteristics. Do not go further than this.)*

**5. Why this matters commercially in his system.** Handing CosyVoice a whole long sentence costs
~3.0s before the first audio sample. Handing it a short opening clause costs ~1.7s. So the system
deliberately splits the first clause off and synthesises it early. That single trick is worth ~1.3s
of felt latency. **Prefill cost is the reason that trick exists.**

**6. From scores to a token: sampling.** The model outputs a score (logit) per vocabulary item. Turn
those into probabilities (softmax). Now — which one do you actually pick?

- **Greedy**: always take the highest. Deterministic.
- **Temperature**: flatten or sharpen the distribution before choosing.
- **Top-k**: sample among the k likeliest.
- **Top-p (nucleus)**: sample among the *smallest set whose probabilities sum to p* (e.g. 0.9). The
  set size adapts to how confident the model is. This is the modern default.
- **Repetition penalty**: down-weight tokens that recently appeared.

**7. THE PUNCHLINE — degenerate repetition.** Greedy decoding can enter a **cycle it can never
leave**: it picks token A, which makes B likeliest, which makes A likeliest again — forever. Because
the model's output is fed back as its input, a self-reinforcing state is a *trap*.

His toy shows it, verbatim: greedy generation locks into
`(`tailscale (`tailscale (`tailscale…` and never escapes.

Top-p usually escapes — but *probabilistically*, not reliably. A **repetition-aware sampler**
(down-weight anything emitted recently) escapes **by construction**.

Have him notice the tuning subtlety in the toy: with a look-back window of 12 the tight loop dies but
a *longer* cycle forms outside the window; with 40, over-penalising creates new attractors. 24 works.
**Sampling policy is a real engineering surface, not a cosmetic setting.**

**8. How a neural TTS works (enough of it).** Text → an autoregressive model emits *semantic/acoustic
tokens* (exactly like an LLM emitting words) → a decoder/vocoder turns those tokens into an actual
waveform. So a TTS has **the same prefill/decode/sampling behaviour as an LLM**, because at its core
it *is* one. This is the connection that makes his system make sense.

Also mention **zero-shot voice cloning**: give the model a few seconds of reference audio and it
conditions on that speaker, no retraining. That's how his avatar has a specific person's voice.

**9. What vLLM is.** A high-performance inference server (continuous batching, paged attention). It
made his TTS dramatically faster. **Remember that it is a different implementation of the same model** —
which is what causes the bug below.

## The blanks he must fill (DO NOT write these for him)

1. Greedy: return the highest-count next character.
2. Top-p: accumulate the sorted probabilities until they reach p; keep that nucleus.
3. Top-p: sample one item from the nucleus, **weighted** by its count.
4. Repetition-aware: multiply the weight of any recently-seen character by a penalty, then take the max.

## Misconceptions to hunt

- *"A longer answer means I wait longer for the first word."* Backwards — it's the *input* length.
- *"Greedy gives the best quality because it always picks the most likely token."* It is the mode most
  prone to degenerate looping. Locally optimal, globally broken.
- *"Sampling settings are a style preference."* In his system, losing one sampling rule turned a
  4-second sentence into 12 seconds of dead silence. It is a **correctness** feature.
- *"A faster implementation of the same model behaves the same."* This is the big one. See below.

## The bug this explains — "P18"

His team moved the TTS's internal language model onto **vLLM** for speed. Huge win. But vLLM's
sampling pipeline **silently dropped the repetition-aware sampler** that the original implementation
had.

The result: Chinese synthesis intermittently **looped on the silence token**. A 4-second sentence
became ~12 seconds of dead air. It was heard as "halting" speech — and the avatar kept moving its
mouth through the silence, because the mouth-generator follows the waveform it's given.

The fix was to reimplement the repetition-aware sampler as a vLLM logits processor.

**The shape of this bug is the lesson, and it recurs:** *an optimisation silently removed a
correctness guarantee that nobody had written down.* It appears again in his project as "P33", where
enabling CUDA graphs made the benchmark faster and the **lip-sync worse** — and the human eye
overruled the stopwatch. Get him to articulate why that's the same bug wearing a different hat.

## Exit criteria

- Explains why TTFT scales with input, not output, and why that justifies the first-clause split.
- Explains why greedy decoding can loop and what repetition-aware sampling does about it.
- Explains why swapping in a faster inference engine is a **correctness** risk, not just a speed change.

---

# MODULE 4 — Pixels, GPUs, and the sync contract

**Toy:** `learn/m4_sync.py` · **4 blanks** · **The bugs it explains: "P16" and "P1"**

## Objective

He understands why two models on one GPU contend, what a talking-head model does per frame, what
TensorRT and CUDA graphs actually are — and, above all, the **audio/video sync contract**: that when
a renderer can't keep up, someone must pay, and engineering is choosing *who*.

## Teach, in this order

**1. What a GPU is, minimally.** Thousands of small cores doing the same operation on lots of data.
A **kernel** is a function launched onto it. Data must be copied to **VRAM** (the GPU's own memory)
before it can be used. VRAM is small and fixed — his card has 16 GB.

**2. Contention.** His TTS and his avatar model are **two separate processes on one card**. They both
want VRAM (if either takes too much, the other fails to start) and they both want compute (so the GPU
time-slices between them and *both get slower*). This one fact generates a large fraction of his
project's bug list.

Consequence worth stating plainly: **the avatar's real frame rate is a variable, not a constant.** It
depends on what the TTS is doing at that moment.

**3. Frames, fps, frame budget.** Video is still images at a fixed rate. At 14 fps you have
1/14 = **71 ms** to produce each frame. Miss that budget and you are behind — and you do not get the
time back.

**4. What the talking-head model does per frame** (behavioural, not architectural):
- Once, at startup: find the face in the source portrait, and locate the mouth region.
- Per frame: take a window of the **audio**, run a speech model over it to get features that encode
  *what sound is being made*; feed those features plus the masked mouth region into a neural network
  that generates the mouth; decode it back to pixels; composite it onto the original face.

The one consequence he must extract: **the mouth is generated from the audio waveform.** So if the
waveform is garbage, the mouth is garbage — *this is why Module 1's bug destroyed the lip-sync.* Make
him connect these himself; it is the moment the whole course closes into a loop.

**5. TensorRT.** An ahead-of-time compiler for neural networks: it fuses operations, picks optimal
kernels for the specific GPU, and uses lower precision (fp16). You compile once, into an engine file,
and it runs much faster. Cost: the engine is tied to that GPU and to fixed input shapes. In his
project it cut render time per segment from ~389 ms to ~255 ms.

**6. CUDA graphs.** Normally the CPU launches every kernel one at a time, and that launch overhead
adds up. A CUDA graph **captures** a whole sequence of launches once and replays it as a single unit.
Faster — but capture fixes the execution, and **can subtly change numerical behaviour.** Hold that
thought.

**7. THE CENTRAL LESSON — the A/V sync contract.**

Set up the conflict crisply. Audio plays at exactly one second per second; physics will not negotiate.
Video is produced by a renderer whose speed is a *variable*. Target 14 fps, but under GPU contention
you actually get 10.

So you are producing 10 frames for every 14 you need. **You are 4 frames per second in debt, and the
debt is never repaid.** Ask him: after a 20-second reply, how far behind are the lips?

Have him derive it: `drift = turn_length × (1 − real_fps / target_fps)`. His toy measures it:
**0.87s at a 2s turn, 3.27s at 8s, 8.07s at 20s.** Dead linear.

**This explains why the bug hid for so long: short replies looked completely fine.**

Now the choice. There are exactly two options:

- **Audio-master (`live`)**: play the voice immediately at real time; show video whenever it arrives.
  Nothing ever pauses. The lips slide progressively behind the words. *(Note for context: this is what
  conventional media players do — they use audio as the master clock and drop video frames, because
  humans notice an audio glitch far more than a dropped frame. His system is unusual in rejecting it,
  and it is worth asking him why: a talking face whose mouth doesn't match is a special kind of wrong.)*
- **Video-master (`steady`)**: pin audio frame N to video frame N and release them together. Drift
  becomes **structurally impossible**. But the voice must now *wait* for the renderer — so a render
  stall becomes a **silence**.

He chose video-master: *a pause is survivable; lips that slide off the words are not.*

**8. The twist he must not miss.** Video-master alone is **not enough**. If the renderer is
*permanently* below target, it doesn't pause occasionally — **it stretches the entire voice**. His toy
shows this brutally: every single one of 111 audio gaps exceeds the frame budget. It looks like the
mode is broken. It isn't. That is simply what a permanently-too-slow renderer does to a video-mastered
stream.

Run it with TensorRT (render now *above* target) and it becomes **0 of 111 gaps**.

**So the real insight — make him say it — is: TensorRT did not fix the sync logic. The sync logic was
correct all along. It bought back the *headroom* that made the correct logic free.** That is a
different and much more useful way to think about performance work.

## The blanks he must fill (DO NOT write these for him)

1. `live`: drift = when the frame was rendered − when its audio should have played (`i / target_fps`).
2. `steady`: audio for frame i can only be released once frame i exists — so its release time *is* the
   render time.
3. `steady`: drift is zero by construction; the cost reappears as **gaps** between consecutive
   releases. Return those gaps.
4. Write **one sentence, in his own words**, explaining why drift *grows* with turn length rather than
   staying constant. This one is the real exam — do not accept a vague answer.

## Misconceptions to hunt

- *"The GPU is fast, so rendering is basically free."* Not when two models share it.
- *"Higher resolution = a better avatar."* His team measured the opposite: pushing resolution starved
  the renderer, which caused voice lag. Quality and latency are coupled through the shared GPU.
- *"The benchmark got faster, so it got better."* The CUDA-graphs case: faster stopwatch, worse
  lip-sync. **The eye overruled the benchmark, and the eye was right.**
- *"Sync is a bug to fix."* No — sync is a *contract you choose*. Both options are correct; they
  differ in who pays.

## The bugs this explains

**"P16"** — the lips drifted further behind the longer the reply got, because the render rate was
below target and the deficit accumulated. Fixed with TensorRT: not by correcting the sync, but by
buying the headroom it needed.

**"P1"** — a single flag (`cudnn.benchmark = True`) made the library re-tune its algorithm whenever
the input shape changed. The first segment of every turn has a different shape than mid-turn ones. So
**every single turn** ate a ~16-second GPU spike. One boolean. Ask him what class of bug that is
(answer: an optimisation that assumed a *stable* workload, in a system whose workload is stable in the
middle and never at the start).

## Exit criteria

- Derives the drift formula and explains why it scales with turn length.
- Explains audio-master vs video-master and argues *for* the choice his system made.
- Explains why TensorRT was load-bearing rather than a nice-to-have.
- Explains why a faster benchmark can mean a worse product.

---

## After the four modules

He returns to `learn/index.html` for the capstone: drive the real system, produce a per-hop latency
waterfall (layer / file / milliseconds / the knob that moves it), and then explain a bug from his
project's history **cold**, before reading the write-up.

**Then — and please reinforce this with him, it is the half people skip — he must come back at week 6
and week 10 and re-answer everything he got wrong.** Spaced practice is not decoration; it is one of
the only two techniques that reliably work. Studied once is not studied. Anything that decays before
week 10 was activity, not learning.

---

## Glossary (for quick reference)

| Term | Meaning |
|---|---|
| **Sample** | One number = the air pressure at one instant. |
| **Sample rate** | Samples per second (16,000 Hz for speech). |
| **int16 / PCM** | Each sample is a signed 16-bit int = **exactly 2 bytes**. Raw audio, no header. |
| **RMS** | Root-mean-square = "how loud is this chunk". |
| **VAD** | Voice activity detection — is anyone speaking right now? |
| **Frame (audio)** | A small chunk of audio, ~20 ms. |
| **Frame (video)** | One image. |
| **Event loop** | The scheduler that interleaves concurrent tasks on one thread. |
| **Backpressure** | A full bounded queue blocking its producer — the system self-regulating. |
| **Little's Law** | `L = λW`. Queue length *is* latency at fixed throughput. |
| **Streaming** | Emitting the first piece before the last one exists. |
| **TTFO / TTFT** | Time to first output / first token — what the user actually waits for. |
| **Autoregressive** | Emits one token, feeds it back, emits the next. |
| **Prefill / decode** | Processing the input (before any output) / emitting tokens one by one. |
| **Top-p (nucleus)** | Sample from the smallest token set whose probabilities sum to p. |
| **RAS** | Repetition-aware sampling — down-weight what was just emitted. |
| **Vocoder** | Turns model-generated audio tokens into an actual waveform. |
| **vLLM** | A fast inference server for autoregressive models. |
| **VRAM** | The GPU's own memory. Small, fixed, contended. |
| **TensorRT** | Ahead-of-time compiler producing a fast, GPU-specific engine. |
| **CUDA graph** | A captured, replayable sequence of GPU kernel launches. |
| **Drift** | How far video has fallen behind audio. Grows with turn length. |
| **Audio-master / video-master** | Who waits for whom when the renderer can't keep up. |
