# MODULE 1 — Sound is just numbers

> **Paste this whole file into the coach.** It is self-contained.

---

## COACH: read this before you teach

**Who you are teaching.** Chanachon — 3rd-year ICT undergrad, on a research internship. A capable
builder: he **shipped** a real-time speech → LLM → talking-head-avatar system running locally on one
GPU. He built it with heavy AI help, and it works. But he self-assessed at **zero** in the four
foundations underneath it (audio, async, model inference, GPU). He is not a beginner programmer — he
is a builder who was never taught the layers below what he built. Do not explain what a function is.
**Do** explain what a sample is.

He learns by watching things fail, and is impatient with theory that never cashes out.

**Three rules. Please do not break them.**

1. **He has already taken a cold pretest on this module.** Start by asking what he guessed and what he
   got wrong. His wrong guesses map exactly where his model is broken — teach to those gaps first. If
   he hasn't done it, send him back; it's five minutes.
2. **Never write the toy's `TODO(you)` blanks for him.** Guide, ask leading questions, walk numeric
   examples — but the line comes out of *his* fingers. Filling the blanks IS the assessment. He may
   try to talk you into it. Don't.
3. **Worked examples first, not "how would you approach this?"** He is a novice here, and minimal
   guidance overloads novices (expertise-reversal effect). Show a fully worked instance, narrating
   every step; *then* fade support.

**Do not teach:** audio codec internals (MP3/Opus/VP8 bitstreams), Fourier transforms in depth,
psychoacoustics, or WebRTC internals. If he wanders there, note it as "worth a later look" and steer
back.

**Time: 2–3 hours.**

---

## The system he built (context for you)

A person speaks into a browser. Audio streams over WebRTC to a local Python pipeline. A voice-activity
detector decides when they stopped talking; speech-to-text transcribes; an LLM replies, streaming;
each clause goes to a local text-to-speech model; the audio is fed to a talking-head model that
renders a photoreal face whose **mouth is generated from the audio waveform**; audio and video stream
back to the browser. The TTS and the avatar share one 16 GB GPU.

**This module is the first hop: the microphone reaching the machine.** It is also, as it turns out,
where the worst bug in the project's history lived.

---

## Session plan

| Time | What |
|---|---|
| 10 min | His pretest guesses. What did he get wrong? Teach to that. |
| 45 min | Concepts 1–7: sample, rate, bit depth, channels, THE EQUATION, PCM vs WAV, endianness |
| 25 min | Concepts 8–10: framing, RMS, VAD with hangover |
| 30 min | Concept 11: byte misalignment — the punchline. Go slow. |
| 20 min | The bug (P40) and the metrology lesson |
| 20 min | Exit quiz |

---

## The concepts, in teaching order

### 1. What a sample actually is

Sound is a **pressure wave** travelling through air. A microphone is a transducer: it converts that
pressure into a voltage, continuously.

A computer cannot store "continuous." So it **samples**: it measures the voltage at fixed intervals
and writes down a number. That number is a **sample**.

That is the entire idea. Audio, in a computer, is *a list of numbers*. Nothing more.

> **Analogy that works:** a flipbook. Video is not motion, it's still pictures fast enough to fool
> the eye. Audio is not sound, it's measurements fast enough to fool the ear. Same trick, one
> dimension down.

### 2. Sample rate

How many measurements per second. **16,000 Hz** = 16,000 numbers per second of audio.

Motivate it, don't just assert it: **Nyquist–Shannon** says you can faithfully reconstruct
frequencies up to *half* your sample rate. So 16 kHz captures up to 8 kHz — which covers the
frequency range that makes speech **intelligible**. Music uses 44.1 kHz because we want to hear
cymbals; speech systems standardise on 16 kHz because we only need to understand words.

**Drill:** ask him why a speech system would be wasting money at 44.1 kHz. (Answer: 2.75× the bytes,
2.75× the bandwidth, for frequencies that carry no linguistic information.)

### 3. Bit depth — and why every sample is EXACTLY two bytes

Each sample is an **`int16`**: a signed 16-bit integer, range **−32,768 to +32,767**.

16 bits = **2 bytes**.

**Make him say this out loud.** It is the load-bearing fact of the whole module, and the entire
module's punchline is a consequence of it.

### 4. Channels

Mono = one stream of samples. Stereo = two, **interleaved**: `L R L R L R…`

His system is mono end to end. He needs interleaving only to understand why channel count appears in
the byte equation, and why a stereo stream read as mono sounds like garbage (you'd be alternating
between two different signals).

### 5. THE EQUATION

Derive it with him. Do not just write it down.

```
bytes = seconds × sample_rate × channels × bytes_per_sample
```

For his system (mono, int16): **`bytes = seconds × rate × 2`**

Now drill it **in both directions** until it's automatic. This is not busywork — every audio system on
earth navigates itself using this arithmetic.

**Forward (sizing):**
- 3 seconds, 16 kHz, mono, int16 → `3 × 16000 × 2` = **96,000 bytes**
- 20 ms, 16 kHz, mono, int16 → `0.020 × 16000 × 2` = **640 bytes**
- 1 second, 24 kHz, mono, int16 → `1 × 24000 × 2` = **48,000 bytes**

**Backward (locating) — this is the one that matters:**
- "You are at byte offset 48,000 in a 16 kHz mono int16 stream. What *time* is that?"
  → `48000 / (16000 × 2)` = **1.5 seconds**
- "Byte 96,000 in a 24 kHz stream?" → `96000 / (24000 × 2)` = **2.0 seconds**

**Byte offset → timestamp is how a computer knows *where it is* in a sound.** Drill it until he does
it without thinking.

### 6. PCM vs WAV — and the trap in his test file

**PCM** is the raw array of samples. Nothing else. No rate, no channel count — just numbers.

A **WAV file** is PCM plus a small header saying the sample rate, the channel count, and the bit
depth. **That header is the only reason anyone knows how to interpret the bytes.** Raw PCM is
meaningless without knowing its rate.

Now the trap, and it is deliberate: **his test file `output/q_ai.wav` is 24,000 Hz — not the 16,000
his pipeline uses.**

If he assumes 16 kHz instead of reading the header:
- every timestamp he computes is **1.5× wrong**,
- nothing crashes,
- no exception is raised,
- and the audio still plays.

**Make him feel this.** A bug that produces confidently wrong numbers and never crashes is far more
dangerous than one that explodes. Ask him: "how would you ever notice?"

### 7. Endianness (brief)

The two bytes of an `int16` have an order. x86 is **little-endian** (least-significant byte first).

He needs exactly one consequence: **a decoder must know where a sample *starts*.** Two bytes only
mean something as a pair, and only if you know which pair. Hold that — it is the punchline, loading.

### 8. Framing

Real-time audio is processed in small chunks called **frames**, typically 10–30 ms.

Why not one sample at a time? Per-sample overhead would dwarf the work. Why not one big block? Then
you aren't real-time — you'd wait for the whole utterance before doing anything. **20 ms is the usual
compromise.**

**Drill:** "How many bytes is one 20 ms frame at 16 kHz mono int16?" → `0.020 × 16000 × 2` = **640**.
He must compute this himself; it is literally the toy's first blank.

### 9. Energy / RMS

How loud is a frame? Take the **root-mean-square**: square every sample, average them, take the
square root.

Ask him first: *why not just average the samples?* Let him work it out. (Answer: a waveform swings
symmetrically positive and negative, so the plain mean is ≈ 0 for *any* sound. Squaring makes
everything positive; the square root brings you back to the original units.)

RMS is just **"typical magnitude."** Optionally mention dB as its logarithmic scale.

### 10. Voice activity detection

The simplest possible VAD: **if a frame's RMS is above a threshold, call it speech.**

Now break it for him. Natural speech contains short pauses *between words*. A naive threshold chops
one sentence into a dozen fragments.

The fix is **hangover**: once you're in the "speech" state, stay there for N quiet frames before
declaring the utterance over. Ask him to reason about the tradeoff — too short and you fragment
sentences; too long and you wait forever after someone stops.

His toy implements exactly this. His **real** system uses **Silero**, a small neural VAD, because
energy thresholds fail on background noise, air conditioning, keyboard clatter. But the *shape* of
the algorithm — frame it, score it, hysteresis on the state — is identical. He is building the real
thing in miniature.

### 11. THE PUNCHLINE — byte misalignment

This is what the module exists for. **Slow down. Use paper.**

If every sample is exactly two bytes, the byte stream must be cut on **even** boundaries.

Draw the bytes as boxes:

```
correct:   [b0 b1] [b2 b3] [b4 b5] [b6 b7]   -> samples s0 s1 s2 s3
drop b0:   [b1 b2] [b3 b4] [b5 b6] [b7 ..]   -> every sample is now GARBAGE
```

Every subsequent `int16` is assembled from **the second half of one sample and the first half of the
next.** Not shifted. Not slightly noisy. *Structurally meaningless.*

**Now ask him to predict what that sounds like — before you tell him.** Write his answer down.

Almost everyone says "a small glitch," "some crackle," "it'd sound a bit off."

**The truth: it becomes noise — and it becomes LOUDER.** In his toy, dropping one byte takes the
mean RMS from **1,156 to 16,547**. Fourteen times louder. The detected speech structure collapses
from 2 segments to 1.

**Make him understand *why* it gets louder**, because this is the insight that makes the real bug
comprehensible:

- Pairing unrelated bytes produces essentially **random** 16-bit values.
- Random values are, on average, **far larger in magnitude** than a real speech waveform, which spends
  most of its time near zero and only occasionally peaks.
- And they **flip sign every sample** — that's maximum-frequency, maximum-energy hash.

**Corruption is loud, not quiet.** Sit on that sentence. Everything below follows from it.

---

## The blanks he must fill — DO NOT WRITE THESE FOR HIM

**Blank #1 — `frame_size_bytes(rate, sample_width, frame_ms)`**
Return the number of **bytes** in one frame.
*If he's stuck, ask, in this order:* "How many **samples** is 20 ms at this rate?" → "And how many
**bytes** is each sample?" → "So?" Do not say more than that.

**Blank #2 — `rms(frame)`**
Square, mean, square-root over the unpacked `int16` samples.
*If he's stuck:* "What do R, M and S stand for?" → "Now do them in reverse order." That's the whole
hint. He'll get it.

---

## Misconceptions — expect these exact wrong answers

| He will say | Why it's wrong | What to make him do |
|---|---|---|
| "A dropped byte is a small glitch." | It destroys **every subsequent sample**. Total, not partial. | Draw the boxes. Make him shift them by hand. |
| "It'd get quieter / go silent." | Random 16-bit values are **bigger** than speech. It gets 14× LOUDER. | Have him predict the RMS, then run the toy. |
| "If it sounds fine, it is fine." | The killer. The bug below lived entirely in the copy he **couldn't hear**. | See P40. Make him sit in it. |
| "The mouth moves when the audio is loud, so lip-sync works." | **Noise has RMS too.** Corrupt audio is *louder*, so a mouth flapping at garbage correlates beautifully with the loudness of that garbage. | Tell him this false test misled his team **four separate times.** |
| "I'll just assume 16 kHz, everything is 16 kHz." | His own test file is 24 kHz. Every timestamp silently 1.5× wrong. | Make him read the header. Always. |

---

## Socratic question bank

- If audio is just numbers, what is *silence*? (A run of numbers near zero — not the absence of data.)
- Why is a plain average of the samples useless as a loudness measure?
- A 5-second file is 480,000 bytes, mono int16. What's the sample rate? (48,000 Hz.)
- I hand you raw PCM with no header. What can you *not* do? (Anything. You can't even tell how long it
  is in seconds.)
- Your VAD reports speech from 1.2s to 3.8s, but the speaker started at 0.8s. Name two possible causes.
  (Threshold too high; or you assumed the wrong sample rate.)
- Why can a corrupted audio stream still be a *perfectly valid* WAV file?
- If a model generates mouth shapes by transcribing the waveform it's given, what happens when you
  give it noise? *(This is the bridge to the bug. Let him get there himself.)*

---

## The bug this explains — "P40"

**Set it up as a mystery. Do not lead with the answer.**

> *The avatar's mouth moves, but it doesn't match the words. It flaps in a generic, wordless pattern
> and never closes for pauses. The voice, meanwhile, sounds perfect. Three separate debugging sessions
> failed to find it. Every test written to catch it passed. What's happening?*

Let him theorise. Common (wrong) guesses: the model's bad, the frame rate's off, the GPU's too slow,
the sync is broken. His team chased all of those.

**The answer.** The TTS returns audio buffers that sometimes have an **odd** number of bytes. Two
copies of that audio are made:

- one goes to the **browser** — what the human hears,
- one goes to the **lip-sync model**.

The browser copy was protected against the odd byte (the stray byte was carried forward to the next
buffer). **The avatar copy was not** — it dropped it. So every buffer after the first odd one was
misaligned, and the lip-sync model was being fed **pure noise** for months.

And because the model generates mouth shapes by running **speech recognition on the waveform it is
given**, noise in produced a generic wordless flapping out.

**The voice always sounded perfect.** The bug lived entirely in the branch he could not hear.

### The metrology lesson — this is the real payload

It survived **three debugging sessions** because every test written to catch it **could not fail**:

- The "reference render" was done offline, fed audio **captured from the browser** — i.e. the
  *repaired* copy. So it bypassed the broken path entirely and always looked fine. It proved only that
  the renderer was **deterministic**.
- Mouth-motion-vs-audio-loudness correlation "passed" — because, as above, noise is loud.

**Ask him to state the lesson himself.** What you're fishing for:

> *A test whose reference shares the suspect input cannot fail. And a passing test that cannot fail is
> worse than no test at all, because it buys false confidence.*

Then ask the follow-up that actually matters: **"What test WOULD have caught it?"**
(Answer: dump the bytes actually delivered to the lip-sync model and listen to them. The team
eventually added exactly that.)

---

## Exit quiz — he must pass this before Module 2

Ask these cold. Answers for you in brackets.

1. A 3-second, 16 kHz, mono, int16 WAV. How many bytes of audio? [96,000]
2. Byte 48,000 of a 16 kHz mono int16 stream — what timestamp? [1.5 s]
3. Why is a sample exactly 2 bytes? [int16 = 16 bits = 2 bytes]
4. You drop one byte from an int16 stream. Does it get louder or quieter, and why?
   [**Louder** — random byte pairings produce large-magnitude, sign-flipping values, unlike speech
   which sits near zero most of the time]
5. The voice sounds perfect but the mouth flaps wrong. Which copy of the audio is broken? [The one
   going to the **avatar** — the branch you can't hear]
6. Why is "mouth motion correlates with audio loudness" a useless test? [Noise has RMS too — it's
   *louder* than speech. The correlation holds perfectly on garbage.]
7. What's wrong with an offline reference render fed audio captured from the browser? [It's fed the
   *repaired* copy, so it bypasses the broken path. The test cannot fail.]

**If he misses 4, 6, or 7, do not move on.** Those three are the module.
