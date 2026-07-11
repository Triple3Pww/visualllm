# Module 1 — Sound is just numbers

## Who I am and what I'm doing

I'm Chanachon, a 3rd-year ICT student at Mahidol, currently on a research internship. I built a
real-time system where you speak into a browser and a photoreal talking-head avatar answers you out
loud, live. The chain is: microphone → detect speech → speech-to-text → LLM → text-to-speech →
a model that animates a face whose mouth matches the words → back to the browser. It runs locally on
one GPU and it works — but I built it with heavy AI help and I never learned the fundamentals
underneath it. I'm fixing that, one layer per week.

**This module is the first hop: audio getting from the mic into the machine.** Teach me this content,
then quiz me with the questions at the bottom. I have a small Python exercise of my own to do after,
so please don't write code for me — explain the ideas.

---

## What I need to learn

### Audio is a list of numbers
Sound is a pressure wave. A microphone turns it into a voltage. A computer can't store "continuous," so
it **samples**: measures the voltage at fixed intervals and writes down a number. That number is a
**sample**. Audio in a computer is just an array of numbers — like a flipbook is just still images.

### Sample rate
How many samples per second. Speech systems use **16,000 Hz** (16 kHz). The reason: the Nyquist theorem
says you can faithfully capture frequencies up to *half* the sample rate, so 16 kHz covers up to 8 kHz —
enough for speech to be intelligible. Music uses 44.1 kHz because we want to hear cymbals; speech
doesn't need it.

### Bit depth — why a sample is exactly 2 bytes
Each sample is an **int16**: a signed 16-bit integer (−32,768 to +32,767). 16 bits = **2 bytes**. This
one fact is the whole module.

### Channels
Mono = one stream of samples. Stereo = two, interleaved (L, R, L, R…). My system is mono throughout.

### The equation (learn it both directions)
```
bytes = seconds × sample_rate × channels × bytes_per_sample
```
For my system (mono int16): **bytes = seconds × rate × 2**.
- Forward: 3 s at 16 kHz mono int16 = 3 × 16000 × 2 = **96,000 bytes**.
- Backward: byte offset 48,000 in a 16 kHz mono int16 stream = 48000 / (16000 × 2) = **1.5 seconds**.

That backward direction — byte offset → timestamp — is how any audio system knows *where it is* in a
sound. I want to be fast at it.

### PCM vs WAV
**PCM** is the raw array of samples, nothing else. A **WAV file** is that array plus a small header
saying the rate, channels, and bit depth. The header is the only reason anyone knows how to interpret
the bytes. (Trap to warn me about: one of my test files is actually 24 kHz, not 16 kHz — if I assume
the rate instead of reading the header, every timestamp I compute is 1.5× wrong and nothing crashes.)

### Framing
Real-time audio is processed in small chunks called **frames**, ~20 ms each. Not one sample at a time
(too much overhead), not the whole utterance (not real-time anymore). One 20 ms frame at 16 kHz mono
int16 = 0.020 × 16000 × 2 = **640 bytes**.

### Loudness / RMS
How loud is a frame? **Root-mean-square**: square every sample, average, square-root. Why not a plain
average? A waveform swings equally positive and negative, so its average is ≈ 0 for any sound. Squaring
makes it all positive; the root brings it back to normal units. RMS = "typical magnitude."

### Voice activity detection (VAD)
Simplest version: if a frame's RMS is above a threshold, call it speech. Problem: natural speech has
short pauses between words, so a naive threshold chops one sentence into fragments. Fix = **hangover**:
stay in the "speech" state for a few quiet frames before declaring it over. (My real system uses a small
neural VAD called Silero because energy thresholds fail on background noise, but the shape is identical.)

### The punchline — byte misalignment
Because every sample is exactly 2 bytes, the byte stream must be cut on **even** boundaries. Drop or
insert a **single byte**, and every following int16 is assembled from the second half of one sample and
the first half of the next. The result is still a valid WAV with plausible loudness — but it's **noise**,
and here's the surprising part: it's **louder**, not quieter. Random byte-pairings produce large,
sign-flipping values, whereas real speech sits near zero most of the time. In my exercise, one dropped
byte takes the loudness (RMS) from ~1,156 to ~16,547 — about 14× louder — and the speech structure
collapses. **Corruption is loud, not quiet.** That sentence is the key to the bug below.

---

## The bug this explains ("P40" in my project)

My avatar's mouth was moving but not matching the words — a generic flap that never closed for pauses —
while the voice sounded perfect. It took three debugging sessions to find.

The cause: my text-to-speech sometimes returns audio buffers with an **odd** number of bytes. Two copies
of the audio are made — one for the browser (what I hear), one for the lip-sync model. The browser copy
carried the odd byte forward correctly. The avatar copy **dropped it**, so the lip-sync model was fed
misaligned noise. And because that model generates mouth shapes by running speech recognition *on the
waveform*, noise in → generic flapping out. The voice sounded perfect because that was the *other* copy.

Why it hid for three sessions: every test written to catch it *couldn't fail*. The offline "reference"
render was fed audio captured from the browser — the *repaired* copy — so it always looked fine. Lesson:
**a test whose reference shares the suspect input cannot fail, and a passing test that can't fail is
worse than no test.**

---

## Questions for Gemini to ask me

1. A 3-second, 16 kHz, mono, int16 WAV — how many bytes of audio?
2. I'm at byte 48,000 in a 16 kHz mono int16 stream. What timestamp is that?
3. Why is a sample exactly 2 bytes?
4. How many bytes is one 20 ms frame at 16 kHz mono int16?
5. Why can't I use a plain average of the samples to measure loudness?
6. I drop one byte from an int16 stream. Does it get louder or quieter — and why?
7. What's the difference between PCM and a WAV file? What breaks if I assume the wrong sample rate?
8. My voice sounds perfect but the mouth flaps wrong. Which copy of the audio is broken?
9. Why is "the mouth moves when the audio is loud, so lip-sync works" a useless test?
10. What kind of test *would* have caught the byte-drop bug?

If I miss 6, 9, or 10, make me work through it again — those are the heart of this module.
