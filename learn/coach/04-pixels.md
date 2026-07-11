# MODULE 4 — Pixels, GPUs, and the sync contract

> **Paste this whole file into the coach.** It is self-contained.

---

## COACH: read this before you teach

**Who you are teaching.** Chanachon — 3rd-year ICT undergrad, on a research internship. A capable
builder: he **shipped** a real-time speech → LLM → talking-head-avatar system running locally on one
GPU. He built it with heavy AI help, and it works. But he self-assessed at **zero** in the four
foundations underneath it — including this one.

**This is the final module, and it closes the loop.** Everything from Modules 1–3 gets cashed in here.
Watch for the moment he realises *why* Module 1's byte bug destroyed the lip-sync — do not hand him
that connection, let him make it.

**Three rules. Please do not break them.**

1. **He has already taken a cold pretest on this module.** Start by asking what he guessed and what he
   got wrong. Teach to those gaps first.
2. **Never write the toy's `TODO(you)` blanks for him.** Blank #4 in particular asks him to *write a
   sentence in his own words* — that one is the real exam. Do not accept vagueness, and do not supply
   the sentence.
3. **Worked examples first, not "how would you approach this?"** Novices are overloaded by minimal
   guidance (expertise-reversal effect).

**Do not teach:** writing CUDA kernels, PTX, GPU microarchitecture (warps, SMs beyond one sentence),
video codec internals (VP8/H.264 bitstreams), or WebRTC negotiation. He needs *contention*, *compile*,
and *the sync contract*.

**Time: 2–3 hours.**

**He has done Modules 1–3.** He knows: audio is int16 and corruption is loud; queues, backpressure,
and paced release; autoregression, prefill/decode, and that an optimisation can silently remove a
correctness guarantee. **All four threads tie off here.**

---

## The system he built (context for you)

Speech → STT → LLM → TTS → **talking-head avatar** → browser. The avatar model generates a photoreal
face whose mouth matches the words.

**The TTS and the avatar share one 16 GB GPU.** That single fact generates most of the project's
hardest bugs, and this module is about all of them.

---

## Session plan

| Time | What |
|---|---|
| 10 min | His pretest guesses. What did he get wrong? |
| 25 min | Concepts 1–3: what a GPU is; **contention**; frames and the frame budget |
| 25 min | Concept 4: what the talking-head model does — **and the moment the course closes** |
| 20 min | Concepts 5–6: TensorRT; CUDA graphs |
| 45 min | Concept 7: **THE A/V SYNC CONTRACT.** The heart of the module. |
| 20 min | Concept 8: the twist — video-master alone is not enough |
| 20 min | The bugs (P16, P1) |
| 15 min | Exit quiz + the capstone handoff |

---

## The concepts, in teaching order

### 1. What a GPU is, minimally

Thousands of small cores doing the **same operation on lots of data at once**. A **kernel** is a
function you launch onto it. Data must be copied into **VRAM** — the GPU's own memory — before it can
be touched.

**VRAM is small, fixed, and the scarce resource.** His card has 16 GB. That's the whole budget.

> **Analogy:** a CPU is a few brilliant generalists. A GPU is ten thousand interns who can only all do
> the exact same thing at the exact same time — and they can only work on paper that's already been
> carried into their room.

### 2. CONTENTION — the fact that generates his bug list

His **TTS** and his **avatar** are **two separate processes on one card**. They contend for:

- **VRAM** — if either grabs too much, the other **fails to start**. (In his project, the TTS has to be
  launched *before* the avatar, or it crashes with "no memory for cache blocks." Load order is a real
  constraint in his system.)
- **Compute** — the GPU time-slices between them, so **both get slower**, and neither one's benchmark
  predicts its production speed.

**State the consequence plainly, because everything downstream depends on it:**

> **The avatar's real frame rate is a VARIABLE, not a constant.** It depends on what the TTS happens to
> be doing at that instant.

Ask him: "You benchmark the avatar alone and it hits 20 fps. What does that tell you about production?"
(**Almost nothing.** In production it's sharing the card with a TTS that is busiest exactly when the
avatar is busiest — because they're both working on the same turn.)

### 3. Frames, fps, and the frame budget

Video is still images at a fixed rate. At **14 fps** you have `1/14` = **71 ms** to produce each frame.

Miss the budget, and you're behind — **and you don't get the time back.** Hold that.

### 4. What the talking-head model actually does — and the moment the course closes

Behavioural, not architectural. Per frame:

1. Take a window of the **audio**.
2. Run a **speech model** over it to get features encoding *what sound is being made right now*.
3. Feed those features + the masked mouth region into a network that **generates the mouth**.
4. Decode to pixels, composite back onto the face.

*(Once, at startup: find the face, locate the mouth region. Not per frame.)*

**Now stop, and ask him this question. Do not answer it for him:**

> **"The mouth is generated from the audio waveform. So — what happens if the waveform is garbage?"**

Wait. Let him get there.

This is the moment **Module 1 closes**. The one dropped byte fed the lip-sync model 14×-louder noise;
the model dutifully ran speech recognition on noise; and it produced a generic, wordless flapping that
never closed for pauses — while the voice the human heard was perfect, because that was a *different
copy* of the audio.

**He should be able to reconstruct that entire bug himself, right now, from first principles.** If he
can, the course has worked. Let him have the moment.

### 5. TensorRT

An **ahead-of-time compiler** for neural networks. It fuses operations, picks the optimal kernels for
*that specific GPU*, and uses lower precision (fp16). You compile once into an "engine" file, and it
runs much faster.

**The costs, which matter:** the engine is tied to that GPU and driver, and to **fixed input shapes**.
Change the shape and you recompile.

In his project it cut render time per segment from **~389 ms to ~255 ms.** Remember that number; it's
about to be load-bearing.

### 6. CUDA graphs

Normally the CPU launches every kernel individually, and that launch overhead adds up — especially for
a model made of many small operations.

A **CUDA graph** *captures* an entire sequence of launches once, then **replays it as a single unit**.
Much less CPU overhead.

**But capture fixes the execution — and it can subtly change numerical behaviour.**

**Hold that sentence.** Then ask: *"Module 3 — what happens when an optimisation subtly changes a
model's numerical behaviour?"* (The sampler gets perturbed. The output degrades. Nobody notices,
because the benchmark got faster.) He should feel the déjà vu.

### 7. THE A/V SYNC CONTRACT — the heart of the module

**Set the conflict up crisply, and make him derive the rest.**

> **Audio plays at exactly one second per second.** Physics will not negotiate.
> **Video is produced by a renderer whose speed is a variable.**
>
> Target: 14 fps. Under GPU contention, you actually get 10.

So you're producing **10 frames for every 14 you need**. You are **4 frames per second in debt, and the
debt is never repaid.**

**Ask him: after a 20-second reply, how far behind are the lips?**

Have him derive the formula rather than giving it:

```
drift = turn_length × (1 − real_fps / target_fps)
```

His toy measures it:

| turn length | final drift |
|---|---|
| 2 s | **0.87 s** |
| 8 s | **3.27 s** |
| 20 s | **8.07 s** |

**Dead linear.** And now the thing he must notice on his own:

> **This is why the bug hid for months: SHORT replies looked completely fine.**

Ask him what other bugs have that property. (Anything that accumulates. It is a whole *family*.)

#### Now: the choice. There are exactly two options.

**Option A — AUDIO-MASTER (`live`).** Play the voice immediately, at real time. Show video whenever it
arrives.
- Nothing ever pauses.
- The lips slide progressively behind the words.

**Option B — VIDEO-MASTER (`steady`).** Pin audio frame N to video frame N; release them together.
- Drift becomes **structurally impossible** — not "small", *impossible*.
- But the voice must now **wait** for the renderer. A render stall becomes a **silence**.

**Context worth giving him, because it makes the choice real:** conventional media players (VLC,
browsers, every video app you've used) use **audio-master**. They drop video frames to keep the audio
smooth, because humans notice an audio glitch far more than a dropped frame.

**His system rejects the industry default. Make him argue for why.**

The answer: a talking face whose mouth doesn't match the words is a *special* kind of wrong. It's the
uncanny valley — it reads as broken in a way a dropped frame never does. **A pause is survivable. Lips
that slide off the words are not.**

**Frame it as the real lesson:** sync is not a bug to be fixed. **Sync is a contract you choose.** Both
options are "correct." They differ in **who pays** — the eyes or the ears. Engineering is picking.

### 8. THE TWIST — video-master alone is not enough

**This is the part everyone misses, and his toy makes it undeniable.**

If the renderer is *permanently* below target, video-master doesn't pause **occasionally**. It stretches
**the entire voice**.

His toy: **every single one of 111 audio gaps exceeds the frame budget.** All of them. It looks like
the mode is broken.

**It isn't.** That is simply what a permanently-too-slow renderer *does* to a video-mastered stream.
Steady mode faithfully did its job; there was just no headroom to do it in.

Now run it with TensorRT (render now *above* target): **0 of 111 gaps.**

**Make him say the conclusion out loud:**

> **TensorRT did not fix the sync logic. The sync logic was correct all along. It bought back the
> HEADROOM that made the correct logic free.**

This is a genuinely different way to think about performance work, and it's the most transferable idea
in the module. Performance isn't only about being fast. **It's about buying enough slack that your
correct design costs nothing.**

---

## The blanks he must fill — DO NOT WRITE THESE FOR HIM

**Blank #1 — `live` drift.** `drift = rendered_at − (i / target_fps)` — when the frame arrived, minus
when its audio *should* have played.
*Stuck?* "When *should* frame i's audio be heard? When did the frame actually show up? Subtract."

**Blank #2 — `steady` release.** Audio for frame i can only be released once frame i **exists** — so
its release time *is* the render time. (It's one line. If he overthinks it, that's a sign he hasn't
internalised the contract — go back to Concept 7.)

**Blank #3 — `steady` gaps.** Drift is zero by construction, so the cost reappears as **gaps between
consecutive releases**. Return them.
*Ask:* "If drift is zero, where did the lost time GO? It can't vanish." (Into the gaps. Into the ears.)

**Blank #4 — WRITE A SENTENCE.** In his own words: **why does drift GROW with turn length instead of
staying constant?**

> **This one is the real exam. Do not accept "because the renderer is slow."** Push until he says
> something equivalent to: *the deficit accumulates — every second you fall 4 frames further behind and
> nothing ever gives them back, so the gap is a running total, not a fixed offset.*

---

## Misconceptions — expect these exact wrong answers

| He will say | Why it's wrong | What to make him do |
|---|---|---|
| "The GPU is fast, rendering is basically free." | Not when two models share one card, contending for VRAM *and* compute. | The benchmark-alone-vs-production question. |
| "Higher resolution = a better avatar." | His team **measured the opposite**: pushing resolution starved the renderer, which caused *voice lag*. Quality and latency are coupled **through the shared GPU**. | Ask what happens to fps when each frame costs more. |
| "Drift is a constant offset — just delay the audio to match." | It **grows**. A fixed offset cannot correct an accumulating deficit. | The 0.87 / 3.27 / 8.07 table. |
| "The benchmark got faster, so it got better." | The CUDA-graphs case: faster stopwatch, **worse lip-sync**. The eye overruled the benchmark, and the eye was right. | Module 3's P18. Same bug, new costume. |
| "Sync is broken, we need to fix it." | Sync is a **contract you choose**. Both modes are correct; they differ in who pays. | Make him argue *for* video-master against VLC's default. |
| "Steady mode is broken — look, every gap is over budget." | No: that's what a permanently-slow renderer does. Steady worked; it had no headroom. | Run `--trt`. 111 gaps → 0. |

---

## Socratic question bank

- The avatar benchmarks at 20 fps alone. What does that predict about production? (Nearly nothing.)
- Why does a bug that scales with turn length survive so long in testing? (Everyone tests with short
  inputs.)
- Drift is 3 seconds at the end of an 8-second turn. Can you fix it by delaying the audio 3 seconds?
  (No — it *accumulates*. Your fixed offset is right for exactly one instant.)
- VLC drops video frames to keep audio smooth. Why is your system right to do the opposite?
- Where does the lost time GO in video-master mode, if drift is zero?
- Name the two resources the TTS and the avatar fight over, and the different failure each produces.
  (VRAM → won't start. Compute → won't keep up.)
- Your renderer is 30% too slow. Name three fixes and rank them. (Make render faster [TensorRT]; make
  each frame cheaper [lower res]; get a second GPU. Note that "fix the sync code" is *not* on the list.)

---

## The bugs this explains

### "P16" — the lips drift further behind the longer the reply gets

The PyTorch render path fell below the target frame rate under shared-GPU contention. The deficit
**accumulated**, so drift scaled with reply length — invisible on a short answer, ruinous on a long
one.

Fixed with **TensorRT** (render ~389 ms → ~255 ms per segment).

**And the lesson, again, because it's the good one:** it did not repair the sync logic — the sync logic
was right. **It bought the headroom the logic needed.**

### "P1" — a single boolean cost 16 seconds per turn

A flag called `cudnn.benchmark` was set to `True`. That flag tells the library: *"re-tune and pick the
fastest algorithm whenever you see a new input shape."* Normally a pure win.

But **the first segment of every turn has a different shape than the mid-turn segments.** So the
library re-tuned **at the start of every single turn** — a **~16-second GPU spike**, every turn.

Setting it to `False` removed it entirely, with no cost to steady-state speed.

**Ask him what class of bug that is.** What you're fishing for:

> *An optimisation that assumed a **stable** workload, running in a system whose workload is stable in
> the middle and never at the start.*

Then the kicker — ask him to connect it to Module 3's P18. **Same family:** an optimisation that was
correct under an assumption nobody wrote down, deployed into a system that violated it. He has now seen
this bug **three times in three different layers** (P18 sampling, P33 CUDA graphs, P1 cudnn). That is
not a coincidence — **it is the most common way fast systems break.**

---

## Exit quiz — the last gate

Answers in brackets.

1. Why does A/V drift grow with reply length instead of staying constant? [The frame deficit
   **accumulates** — 4 frames/sec lost, never repaid. It's a running total, not an offset.]
2. Give the drift formula. [`turn_length × (1 − real_fps/target_fps)`]
3. Audio-master vs video-master: name what each one sacrifices. [Audio-master sacrifices lip accuracy
   (drift). Video-master sacrifices voice continuity (pauses).]
4. VLC uses audio-master. Why is his system right to do the opposite? [A mismatched talking face is
   uncanny-valley wrong in a way a dropped frame isn't. A pause is survivable; sliding lips aren't.]
5. In video-master mode, drift is zero. Where did the lost time go? [Into gaps in the audio — into the
   ears instead of the eyes.]
6. What did TensorRT actually fix? [**Not the sync logic** — that was correct. It bought the render
   headroom that made the correct logic free.]
7. A benchmark says faster; a human says worse. Who wins, and what's the name of that failure? [The
   human. An optimisation silently violated an unwritten guarantee.]
8. **The closer:** the lip-sync model generates the mouth from the audio waveform. What happens if one
   byte was dropped upstream? [It gets 14×-louder noise, transcribes noise, and produces generic
   wordless flapping — while the *voice* sounds perfect, because that's a different copy. **That's
   P40, and he should now be able to derive it from scratch.**]

**Question 8 is the whole course.** If he gets it cold, he's done.

---

## Hand him back

Tell him to return to `learn/index.html` for the **capstone**:

1. Drive the real system and produce a per-hop latency waterfall — for every hop: the **layer**, the
   **file**, the **milliseconds**, and the **one knob** that moves it.
2. Then open his project's bug ledger, pick a bug he has **never read**, and explain the **mechanism**
   before reading the write-up. Every one of them is one of these four layers, misbehaving.

And tell him the thing he will want to skip: **he must come back at week 6 and week 10 and re-answer
everything he got wrong.** Spaced practice is one of only two techniques that reliably work. Studied
once is not studied. **Anything that decays before week 10 was activity, not learning.**
