# VisualLLm Fundamentals Course — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `learn/` — an interactive HTML course plus four faded, runnable Python toys that teach Chanachon the four foundations VisualLLm stands on (audio/PCM, async streaming, model inference, GPU/sync), each cashing out on a real bug from `docs/PROBLEMS-AND-FIXES.md`.

**Architecture:** `learn/index.html` is a single self-contained page (no server, no deps, no build) that **teaches and examines**: it gates every answer behind a committed guess, drives spaced interleaved retrieval, and stores an immutable prediction log in `localStorage`. The four `.py` toys are the **lab**: real, runnable, stdlib-only files with `TODO(you)` blanks whose density rises 20% → 70% across the four weeks. The page **never contains code** — it names the file, the blanks, and the run command. That is the anti-drift rule, and it is enforced by a grep.

**Tech Stack:** System Python 3.11 (stdlib only: `wave`, `struct`, `asyncio`, `math`, `time`, `random`, `argparse`). Vanilla HTML/CSS/JS, no framework, no CDN. Existing repo audio in `output/`.

## Global Constraints

- **No test suite.** `CLAUDE.md`: "There is **no build/lint/unit-test suite** — don't invent one." Verification = run the toy and observe stated output, plus the anti-drift grep. Do not add pytest.
- **Toys are stdlib-only**, system Python 3.11 (`python`), **no GPU, no conda env**. A module must never die on setup.
- **`.py` source is ASCII-only** (repo convention: em-dashes/arrows in `.py` cause encoding pain). Use `--` and `->`. The HTML may use full Unicode.
- **Anti-drift rule (non-negotiable):** `learn/index.html` contains **zero Python code**. It may name files, `TODO(you)` ids, and shell commands. It may not contain `def `, `import `, `asyncio.`, or a Python code block. Enforced by the Task 6 grep.
- **Faded worked examples:** each toy ships nearly complete; only the marked `TODO(you) #N` lines are missing. Blank density by design: `m1_vad.py` ~20%, `m2_pipeline.py` ~40%, `m3_sampler.py` ~55%, `m4_sync.py` ~70%.
- **Audio note:** `output/q_ai.wav` is **24 kHz**, not 16 kHz. Toys must read the WAV header, never assume the rate. This is deliberate — it is Module 1's first lesson.
- Commit after each task. Never `git commit -am` (the repo routinely holds unrelated uncommitted work); always `git add <explicit paths>`.

---

### Task 1: Scaffold `learn/` and its README

**Files:**
- Create: `learn/README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: the `learn/` directory that every later task writes into.

- [ ] **Step 1: Create `learn/README.md`**

```markdown
# learn/ — the VisualLLm fundamentals course

**Open `index.html` in your browser. That's it.**

Four modules, one per week. Each follows one conversation turn through the system and
teaches the layer it crosses: audio -> async streaming -> model inference -> GPU and A/V sync.
Every module ends on a real bug from `docs/PROBLEMS-AND-FIXES.md` that you can now explain.

The page teaches and examines you. The `.py` files here are your lab — you open them in
your editor, fill the `TODO(you)` blanks, and run them:

    python learn/m1_vad.py output/q_ai.wav

Nothing here needs a GPU, a conda env, or an install. Plain Python 3.11 and a browser.

**The page holds no code, on purpose.** It names the file and the blank; the code lives only
in the `.py` file, so the two can never drift apart.

Design + evidence: `docs/superpowers/specs/2026-07-11-visualllm-study-plan-design.md`
```

- [ ] **Step 2: Verify the file exists**

Run: `ls learn/`
Expected: `README.md`

- [ ] **Step 3: Commit**

```bash
git add learn/README.md
git commit -m "feat(learn): scaffold the fundamentals course folder"
```

---

### Task 2: Module 1 toy — `m1_vad.py` (audio, ~20% blank)

Teaches: sampling, int16 = exactly two bytes, bytes<->seconds, RMS, and the one-byte
misalignment that caused **P40**. The `--drop-byte` flag reproduces P40 in one flag.

**Files:**
- Create: `learn/m1_vad.py`

**Interfaces:**
- Consumes: `output/q_ai.wav` (24 kHz mono int16, 2.88 s).
- Produces: nothing consumed by later tasks. `index.html` (Task 7) references the filename
  `learn/m1_vad.py`, the blank ids `TODO(you) #1` / `#2`, and the two run commands below.

- [ ] **Step 1: Write `learn/m1_vad.py`**

```python
"""Module 1 toy -- Sound is just numbers.

A from-scratch voice-activity detector. Two blanks are yours; the rest is given.

Run it:
    python learn/m1_vad.py output/q_ai.wav
    python learn/m1_vad.py output/q_ai.wav --drop-byte

The second command drops ONE byte from the front of the audio. Predict what happens
BEFORE you run it -- that is the whole point of the exercise.
"""
import argparse
import math
import struct
import sys
import wave

FRAME_MS = 20          # one VAD frame
SILENCE_RMS = 500      # below this (in int16 units) we call it silence
HANGOVER_FRAMES = 8    # keep speech "on" this many quiet frames before ending it


def read_pcm(path):
    """Return (raw_pcm_bytes, sample_rate, sample_width_bytes)."""
    with wave.open(path, "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            sys.exit("this toy wants mono int16 audio")
        # NOTE: we ASK the file for its rate. q_ai.wav is 24000, not 16000.
        # Assuming 16k here is the classic bug -- every timestamp would be 1.5x wrong.
        return w.readframes(w.getnframes()), w.getframerate(), w.getsampwidth()


def frame_size_bytes(rate, sample_width, frame_ms):
    """How many BYTES is one frame_ms frame of audio?"""
    # TODO(you) #1
    #   samples in the frame = rate * frame_ms / 1000
    #   each sample costs `sample_width` bytes
    #   return an int (number of BYTES)
    raise NotImplementedError("TODO(you) #1 -- see the hint above")


def rms(frame):
    """Root-mean-square loudness of one frame of int16 PCM."""
    n = len(frame) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack("<%dh" % n, frame[: n * 2])
    # TODO(you) #2
    #   square each sample, take the mean, take the square root, return a float.
    raise NotImplementedError("TODO(you) #2 -- see the hint above")


def find_speech(pcm, rate, sample_width):
    """Walk the audio frame by frame and return [(start_s, end_s), ...]."""
    fsize = frame_size_bytes(rate, sample_width, FRAME_MS)
    bytes_per_sec = rate * sample_width
    segments, quiet, start = [], 0, None

    for i in range(0, len(pcm) - fsize + 1, fsize):
        loud = rms(pcm[i:i + fsize]) >= SILENCE_RMS
        t = i / bytes_per_sec          # byte offset -> SECONDS. this is the whole trick.
        if loud:
            if start is None:
                start = t
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= HANGOVER_FRAMES:
                segments.append((start, t))
                start, quiet = None, 0
    if start is not None:
        segments.append((start, len(pcm) / bytes_per_sec))
    return segments


def mean_rms(pcm, rate, sample_width):
    fsize = frame_size_bytes(rate, sample_width, FRAME_MS)
    frames = [rms(pcm[i:i + fsize]) for i in range(0, len(pcm) - fsize + 1, fsize)]
    return sum(frames) / len(frames) if frames else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--drop-byte", action="store_true",
                    help="drop ONE byte from the front. predict the result first.")
    args = ap.parse_args()

    pcm, rate, sw = read_pcm(args.wav)
    print("file        : %s" % args.wav)
    print("sample rate : %d Hz   (did you assume 16000?)" % rate)
    print("sample width: %d bytes/sample" % sw)
    print("audio bytes : %d" % len(pcm))
    print("duration    : %.2f s   = bytes / (rate * width)" % (len(pcm) / (rate * sw)))

    if args.drop_byte:
        pcm = pcm[1:]
        print("\n!! dropped 1 byte. every int16 is now assembled from the WRONG two bytes.")

    print("\nmean RMS    : %.0f" % mean_rms(pcm, rate, sw))
    segs = find_speech(pcm, rate, sw)
    print("speech segments: %d" % len(segs))
    for s, e in segs:
        print("  %.2fs -> %.2fs  (%.2fs)" % (s, e, e - s))

    print("\n--- what you just proved ---")
    print("Audio is a list of int16 numbers. Each one is EXACTLY two bytes.")
    print("Shift by one byte and every sample becomes garbage -- yet it is still")
    print("a perfectly valid WAV, and the loudness stays plausible. That is P40:")
    print("the avatar's lip-sync model was fed exactly this noise for three sessions,")
    print("while the voice a human heard stayed perfect. Read P40 only after this runs.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify it fails at the first blank**

Run: `python learn/m1_vad.py output/q_ai.wav`
Expected: prints the header lines (24000 Hz, 2 bytes/sample), then
`NotImplementedError: TODO(you) #1 -- see the hint above`.

This failure IS the deliverable: the toy is supposed to arrive incomplete. Do **not** fill
the blanks in — they are Chanachon's.

- [ ] **Step 3: Verify the blanks are solvable (temporary check, then revert)**

Temporarily fill both blanks locally to confirm the toy works end to end:
`frame_size_bytes` returns `int(rate * frame_ms / 1000) * sample_width`;
`rms` returns `math.sqrt(sum(s * s for s in samples) / n)`.

Run: `python learn/m1_vad.py output/q_ai.wav` then `python learn/m1_vad.py output/q_ai.wav --drop-byte`
Expected: the normal run finds **1 or more** speech segments with a mean RMS in the hundreds-to-thousands;
the `--drop-byte` run reports a **visibly higher mean RMS** (the misaligned bytes read as loud noise).
If the drop-byte RMS is not clearly higher, the toy fails to teach P40 — fix it before committing.

Then `git checkout learn/m1_vad.py` is NOT possible (uncommitted) — instead re-insert the two
`raise NotImplementedError(...)` lines exactly as written in Step 1 before committing.

- [ ] **Step 4: Confirm the blanks are back**

Run: `grep -c "NotImplementedError" learn/m1_vad.py`
Expected: `2`

- [ ] **Step 5: Commit**

```bash
git add learn/m1_vad.py
git commit -m "feat(learn): m1 toy -- from-scratch VAD, and P40 in one flag"
```

---

### Task 3: Module 2 toy — `m2_pipeline.py` (async streaming, ~40% blank)

Teaches: the event loop, producer/consumer queues, **backpressure**, and the **paced release
clock** that is `MUSETALK_SYNC_MODE=steady` in miniature.

**Files:**
- Create: `learn/m2_pipeline.py`

**Interfaces:**
- Consumes: nothing (self-contained, no audio file).
- Produces: `index.html` (Task 7) references `learn/m2_pipeline.py`, blanks `#1`–`#3`, and the
  three run commands below.

- [ ] **Step 1: Write `learn/m2_pipeline.py`**

```python
"""Module 2 toy -- Streams, queues, and time.

A 4-stage pipeline, the same shape as pipeline/main.py:
    produce -> transform -> render (SLOW) -> deliver

Run it three ways, predicting each BEFORE you run:
    python learn/m2_pipeline.py unbounded   # what happens to latency?
    python learn/m2_pipeline.py bounded     # what changes?
    python learn/m2_pipeline.py paced       # steady mode, in miniature

Three blanks are yours.
"""
import argparse
import asyncio
import time

N_ITEMS = 20
PRODUCE_INTERVAL = 0.05   # producer is FAST: an item every 50ms
RENDER_COST = 0.12        # renderer is SLOW: 120ms per item. it cannot keep up.
FPS = 8.0                 # the paced-release clock: one item every 1/8 s

T0 = time.monotonic()


def stamp():
    return time.monotonic() - T0


async def produce(out, n):
    """The TTS: hands over items as fast as it makes them."""
    for i in range(n):
        await asyncio.sleep(PRODUCE_INTERVAL)
        # TODO(you) #1
        #   Put (i, stamp()) on the `out` queue -- the item and the time it was made.
        #   Use the ASYNC put, not put_nowait. On a BOUNDED queue that await is the
        #   whole lesson: it blocks the producer when the consumer is behind.
        #   That blocking is called BACKPRESSURE.
        raise NotImplementedError("TODO(you) #1")
    await out.put(None)   # sentinel: no more items


async def render(inp, out):
    """The avatar: slow, and the bottleneck. Nothing here is yours."""
    while True:
        item = await inp.get()
        if item is None:
            await out.put(None)
            return
        i, made_at = item
        await asyncio.sleep(RENDER_COST)   # the GPU, basically
        await out.put((i, made_at))


async def deliver_freerun(inp):
    """Ship each item the instant it is rendered."""
    lags = []
    while True:
        item = await inp.get()
        if item is None:
            break
        i, made_at = item
        lag = stamp() - made_at
        lags.append(lag)
        print("  item %2d delivered at %5.2fs  (age %.2fs)" % (i, stamp(), lag))
    return lags


async def deliver_paced(inp):
    """Release item N at a FIXED clock: t_start + N/FPS. This is `steady` mode."""
    lags = []
    start = None
    n = 0
    while True:
        item = await inp.get()
        if item is None:
            break
        i, made_at = item
        if start is None:
            start = stamp()
        # TODO(you) #2
        #   Compute `deadline` -- the wall-clock time (in stamp() units) at which item
        #   number `n` is ALLOWED out: start + n / FPS.
        raise NotImplementedError("TODO(you) #2")

        # TODO(you) #3
        #   If we are EARLY (stamp() < deadline), sleep the difference.
        #   If we are LATE, do not sleep -- just go. (Never sleep a negative number.)
        raise NotImplementedError("TODO(you) #3")

        lag = stamp() - made_at
        lags.append(lag)
        print("  item %2d released at %5.2fs  (age %.2fs)" % (i, stamp(), lag))
        n += 1
    return lags


async def run(mode):
    maxsize = 0 if mode == "unbounded" else 2   # 0 means INFINITE in asyncio
    q1 = asyncio.Queue(maxsize=maxsize)
    q2 = asyncio.Queue()

    consumer = deliver_paced(q2) if mode == "paced" else deliver_freerun(q2)
    _, _, lags = await asyncio.gather(produce(q1, N_ITEMS), render(q1, q2), consumer)

    print("\nmode          : %s" % mode)
    print("queue maxsize : %s" % ("INFINITE" if maxsize == 0 else maxsize))
    print("worst item age: %.2fs   <-- how stale the oldest delivered item was" % max(lags))
    print("total wall    : %.2fs" % stamp())
    if mode == "unbounded":
        print("\nThe producer ran flat out and the queue swallowed everything. The items")
        print("came out fine -- but LOOK AT THE AGE. By the end you were delivering")
        print("something made seconds ago. An unbounded queue does not fix a slow")
        print("consumer; it HIDES it, and pays in latency. This is why bounding matters.")
    elif mode == "bounded":
        print("\nThe queue filled, so `await q.put()` BLOCKED the producer. It was forced")
        print("to slow to the renderer's speed. Latency stayed flat. That is backpressure.")
    else:
        print("\nItems left on a fixed clock (1/%g s apart) instead of the instant they" % FPS)
        print("were ready. That is MUSETALK_SYNC_MODE=steady: the voice is released paced")
        print("to the frames the renderer actually produced, so audio and video cannot")
        print("drift apart. See local_services/musetalk_video.py.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["unbounded", "bounded", "paced"])
    asyncio.run(run(ap.parse_args().mode))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify it fails at the first blank**

Run: `python learn/m2_pipeline.py unbounded`
Expected: `NotImplementedError: TODO(you) #1`

- [ ] **Step 3: Verify the blanks are solvable (temporary fill, then restore)**

Fill locally: `#1` -> `await out.put((i, stamp()))`; `#2` -> `deadline = start + n / FPS`;
`#3` -> `wait = deadline - stamp()` then `if wait > 0: await asyncio.sleep(wait)`.

Run all three modes. Expected:
- `unbounded`: worst item age grows large (roughly **1.4s or more**) — the producer outran the renderer.
- `bounded`: worst item age is **clearly lower** than unbounded — backpressure held it flat.
- `paced`: items are released on an even ~0.125s cadence.

If `bounded` does not beat `unbounded` on worst-age, the toy fails to teach backpressure — fix before committing.
Then restore the three `raise NotImplementedError(...)` lines exactly as in Step 1.

- [ ] **Step 4: Confirm the blanks are back**

Run: `grep -c "NotImplementedError" learn/m2_pipeline.py`
Expected: `3`

- [ ] **Step 5: Commit**

```bash
git add learn/m2_pipeline.py
git commit -m "feat(learn): m2 toy -- backpressure and the paced release clock"
```

---

### Task 4: Module 3 toy — `m3_sampler.py` (model inference, ~55% blank)

Teaches: prefill vs decode (why first-token latency scales with **input** length), greedy vs
top-p sampling, and the repetition loop that **P18** was — a model looping on one token because
repetition-aware sampling was lost. No torch; a char-level Markov model stands in.

**Files:**
- Create: `learn/m3_sampler.py`

**Interfaces:**
- Consumes: `README.md` (any text file) as its training corpus.
- Produces: `index.html` (Task 8) references `learn/m3_sampler.py`, blanks `#1`–`#4`, and the run commands below.

- [ ] **Step 1: Write `learn/m3_sampler.py`**

```python
"""Module 3 toy -- Models that generate one step at a time.

A char-level Markov "model". It is not a transformer, but it is autoregressive --
it emits ONE token, feeds it back, and emits the next. That is the only property
that matters for understanding CosyVoice's and the LLM's latency and failure modes.

Run, predicting each BEFORE you run:
    python learn/m3_sampler.py prefill    # why does a LONGER prompt start SLOWER?
    python learn/m3_sampler.py greedy     # always take the likeliest char
    python learn/m3_sampler.py topp       # sample from the top-p nucleus
    python learn/m3_sampler.py ras        # greedy, but penalise recent repeats

Four blanks are yours.
"""
import argparse
import random
import sys
import time
from collections import defaultdict

ORDER = 4          # context: the last 4 chars
N_GEN = 300        # chars to generate
TOP_P = 0.9
RAS_WINDOW = 12    # look back this far when penalising repeats
RAS_PENALTY = 0.25 # multiply a repeated char's weight by this


def train(path):
    """Count what character follows each 4-char context. This is the whole 'model'."""
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    model = defaultdict(lambda: defaultdict(int))
    for i in range(len(text) - ORDER):
        model[text[i:i + ORDER]][text[i + ORDER]] += 1
    return model, text


def prefill(model, prompt):
    """Walk the prompt to reach the state the model generates FROM.

    A real LLM does exactly this and it is not free: it must process every token of
    the prompt before it can emit token #1. That is why time-to-first-token scales
    with the INPUT length -- the fact behind COSYVOICE_FIRST_PIECE.
    """
    state = ""
    for ch in prompt:
        state = (state + ch)[-ORDER:]
        time.sleep(0.0004)   # stand-in for the per-token cost of a real prefill
    return state


def greedy(dist, recent):
    """Return the single likeliest next char."""
    # TODO(you) #1
    #   `dist` is {char: count}. Return the char with the highest count.
    raise NotImplementedError("TODO(you) #1")


def top_p(dist, recent):
    """Sample from the smallest set of chars whose probability sums to >= TOP_P."""
    items = sorted(dist.items(), key=lambda kv: -kv[1])
    total = sum(c for _, c in items)

    # TODO(you) #2
    #   Walk `items` accumulating count/total until the running sum reaches TOP_P.
    #   Keep those chars in a list `nucleus` (as (char, count) pairs), then stop.
    raise NotImplementedError("TODO(you) #2")

    # TODO(you) #3
    #   Pick ONE char from `nucleus`, at random, weighted by its count.
    #   Hint: random.choices(population, weights=..., k=1)[0]
    raise NotImplementedError("TODO(you) #3")


def ras(dist, recent):
    """Greedy, but a char seen in the last RAS_WINDOW chars is down-weighted.

    This is the shape of the fix in P18. Running CosyVoice's LLM on vLLM silently
    dropped its repetition-aware sampling, so Chinese looped on the SILENCE token --
    a 4s sentence became 12s of dead air, and the avatar lip-synced through nothing.
    """
    # TODO(you) #4
    #   Build a new {char: weight} where weight = count * RAS_PENALTY if the char is
    #   in `recent`, else count. Then return the char with the highest weight.
    raise NotImplementedError("TODO(you) #4")


def generate(model, state, pick):
    out = []
    for _ in range(N_GEN):
        dist = model.get(state)
        if not dist:
            break
        ch = pick(dist, out[-RAS_WINDOW:])
        out.append(ch)
        state = (state + ch)[-ORDER:]
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prefill", "greedy", "topp", "ras"])
    ap.add_argument("--corpus", default="README.md")
    args = ap.parse_args()

    model, text = train(args.corpus)
    print("trained on %s: %d chars, %d contexts\n" % (args.corpus, len(text), len(model)))

    if args.mode == "prefill":
        for prompt in [text[:20], text[:200], text[:1500]]:
            t = time.monotonic()
            prefill(model, prompt)
            print("prompt %5d chars -> time-to-first-token %.3fs" % (len(prompt), time.monotonic() - t))
        print("\nThe model has not emitted ANYTHING yet -- this is all prefill. A longer")
        print("input costs a later first token, even though the output is the same size.")
        print("That is exactly why COSYVOICE_FIRST_PIECE splits off a short opening clause:")
        print("a 16-word opener cost ~3.0s to first audio, a short one ~1.7s. TTFO 4.6 -> 3.2s.")
        return

    pick = {"greedy": greedy, "topp": top_p, "ras": ras}[args.mode]
    state = prefill(model, text[:ORDER])
    print(repr(generate(model, state, pick)))
    print()
    if args.mode == "greedy":
        print("Greedy always takes the likeliest char -- so it can fall into a cycle and")
        print("repeat forever. A model with no defence against this LOOPS. Now run `ras`.")
    elif args.mode == "topp":
        print("Sampling from the nucleus adds variety, and usually escapes the loop --")
        print("but it is a probabilistic escape, not a guaranteed one.")
    else:
        print("Penalising recent chars breaks the cycle by construction. This is P18:")
        print("restore repetition-aware sampling and the Chinese silence-loop dies.")
        print("See cosyvoice/vllm/ras_logits_processor.py in the cosyvoice repo.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify it fails at a blank**

Run: `python learn/m3_sampler.py greedy`
Expected: `NotImplementedError: TODO(you) #1`

Run: `python learn/m3_sampler.py prefill`
Expected: this one **works without any blanks filled** — three lines showing time-to-first-token
rising with prompt length. (It must, so week 3 opens on a win.)

- [ ] **Step 3: Verify the blanks are solvable (temporary fill, then restore)**

Fill locally: `#1` -> `max(dist, key=dist.get)`; `#2` -> accumulate `c/total` into `nucleus` until
`>= TOP_P`; `#3` -> `random.choices([c for c,_ in nucleus], weights=[n for _,n in nucleus], k=1)[0]`;
`#4` -> `max(dist, key=lambda ch: dist[ch] * (RAS_PENALTY if ch in recent else 1))`.

Run: `greedy`, `topp`, `ras`.
Expected: `greedy` output visibly repeats a short cycle; `ras` output does **not**. If greedy does
not loop on this corpus, raise `N_GEN` or lower `ORDER` until it does — the loop is the lesson.

Then restore all four `raise NotImplementedError(...)` lines exactly as in Step 1.

- [ ] **Step 4: Confirm the blanks are back**

Run: `grep -c "NotImplementedError" learn/m3_sampler.py`
Expected: `4`

- [ ] **Step 5: Commit**

```bash
git add learn/m3_sampler.py
git commit -m "feat(learn): m3 toy -- prefill cost, top-p, and the P18 repetition loop"
```

---

### Task 5: Module 4 toy — `m4_sync.py` (GPU contention + A/V sync, ~70% blank)

Teaches: what happens when the renderer is slower than realtime, and the two ways to release
audio against it — **audio-master** (video free-runs and drifts: `live`) vs **video-master**
(audio paced to real rendered frames: `steady`). This is the system's central design decision,
and by week 4 the fade is heavy enough that Chanachon writes both release loops.

**Files:**
- Create: `learn/m4_sync.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `index.html` (Task 8) references `learn/m4_sync.py`, blanks `#1`–`#4`, and the run commands.

- [ ] **Step 1: Write `learn/m4_sync.py`**

```python
"""Module 4 toy -- Pixels, GPUs, and the sync contract.

The avatar renders at TARGET_FPS on paper. Under GPU contention it actually manages
REAL_FPS. The audio does not care -- it plays at exactly one second per second.

So: what do you do with the video?

    python learn/m4_sync.py live      # audio-master: ship audio now, video lags
    python learn/m4_sync.py steady    # video-master: pace audio to the real frames
    python learn/m4_sync.py steady --contention   # now the GPU gets busy mid-turn

Four blanks are yours -- most of both release loops. By now you can write them.
"""
import argparse
import time

TARGET_FPS = 14.0     # what MUSETALK_FPS claims
REAL_FPS = 10.0       # what the GPU actually delivers while CosyVoice is also running
AUDIO_SECONDS = 8.0   # length of the turn's voice

FRAME_BUDGET = 1.0 / TARGET_FPS
REAL_COST = 1.0 / REAL_FPS


def render_frames(seconds, contention_at=None):
    """Yield (frame_index, wall_clock_when_it_finished_rendering).

    The renderer is simulated, not real -- but it is slower than realtime, which is
    the only property that matters. This function is given; do not change it.
    """
    n = int(seconds * TARGET_FPS)
    t = 0.0
    for i in range(n):
        cost = REAL_COST
        if contention_at is not None and t >= contention_at:
            cost = REAL_COST * 1.6      # CosyVoice starts synthesising. we get slower.
        t += cost
        yield i, t


def play_live(frames):
    """AUDIO-MASTER. The voice plays immediately, at real time. Video arrives when it can.

    Report the DRIFT: how far behind the audio each frame lands.
    """
    drifts = []
    for i, rendered_at in frames:
        # TODO(you) #1
        #   `audio_time` is when this frame's audio SHOULD be heard: i / TARGET_FPS.
        #   `drift` is how late the frame is: rendered_at - audio_time.
        #   Append drift to `drifts`.
        raise NotImplementedError("TODO(you) #1")
    return drifts


def play_steady(frames):
    """VIDEO-MASTER. Audio is released paced to the frames actually rendered.

    The voice WAITS for the renderer, so audio and video cannot drift. The cost is
    that a long stall makes the voice pause.
    """
    released = []
    for i, rendered_at in frames:
        # TODO(you) #2
        #   The audio for frame i may only be released once frame i EXISTS.
        #   So its release time is simply `rendered_at`. Append it to `released`.
        raise NotImplementedError("TODO(you) #2")

    # TODO(you) #3
    #   Drift is now zero by construction -- frame i is pinned to audio i. Instead the
    #   cost shows up as PAUSES. For each consecutive pair in `released`, the gap is
    #   released[k] - released[k-1]. A gap LONGER than FRAME_BUDGET means the voice
    #   had to wait. Return the list of those gaps.
    raise NotImplementedError("TODO(you) #3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["live", "steady"])
    ap.add_argument("--contention", action="store_true",
                    help="the GPU gets busy 3s in (CosyVoice starts talking too)")
    args = ap.parse_args()

    at = 3.0 if args.contention else None
    frames = render_frames(AUDIO_SECONDS, contention_at=at)

    print("target fps : %.0f   (MUSETALK_FPS)" % TARGET_FPS)
    print("real fps   : %.0f   (what the shared GPU actually gives you)" % REAL_FPS)
    print("turn length: %.0fs\n" % AUDIO_SECONDS)

    if args.mode == "live":
        drifts = play_live(frames)
        print("final drift: %.2fs   <-- the lips are this far behind the voice" % drifts[-1])
        print("max drift  : %.2fs" % max(drifts))
        # TODO(you) #4
        #   Print ONE sentence, in your own words, explaining why the drift GROWS with
        #   turn length instead of staying constant. (If you cannot, re-read the numbers
        #   above until you can. This is the module's real exam.)
        raise NotImplementedError("TODO(you) #4 -- write the sentence")
    else:
        gaps = play_steady(frames)
        stalls = [g for g in gaps if g > FRAME_BUDGET * 1.05]
        print("frame budget : %.3fs  (1 / target fps)" % FRAME_BUDGET)
        print("audio gaps   : %d of %d exceeded it" % (len(stalls), len(gaps)))
        print("worst gap    : %.3fs" % (max(gaps) if gaps else 0))
        print("\nDrift is ZERO -- frame i is pinned to audio i, by construction.")
        print("The renderer's slowness did not desync anything; it slowed the VOICE.")
        print("That is the trade `steady` makes, and why the user chose it: a brief")
        print("pause is survivable, lips that slide off the words are not.")
        print("See MUSETALK_SYNC_MODE in CLAUDE.md, and P16 for what made this affordable.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and verify it fails at a blank**

Run: `python learn/m4_sync.py live`
Expected: `NotImplementedError: TODO(you) #1`

- [ ] **Step 3: Verify the blanks are solvable (temporary fill, then restore)**

Fill locally: `#1` -> `drifts.append(rendered_at - i / TARGET_FPS)`; `#2` -> `released.append(rendered_at)`;
`#3` -> `return [released[k] - released[k-1] for k in range(1, len(released))]`; `#4` -> any `print(...)`.

Run: `live`, then `steady`, then `steady --contention`.
Expected: `live` final drift is **large and grows with the turn** (~3s at these constants — the
number that made P16 urgent). `steady` reports **zero drift** and instead a set of audio gaps
exceeding the frame budget. `steady --contention` shows **more/larger gaps** than plain `steady`.

If `live`'s drift does not grow with turn length, the toy fails to teach P16 — fix before committing.
Then restore all four `raise NotImplementedError(...)` lines exactly as in Step 1.

- [ ] **Step 4: Confirm the blanks are back**

Run: `grep -c "NotImplementedError" learn/m4_sync.py`
Expected: `4`

- [ ] **Step 5: Commit**

```bash
git add learn/m4_sync.py
git commit -m "feat(learn): m4 toy -- audio-master vs video-master, and where drift comes from"
```

---

### Task 6: The course engine — `learn/index.html` (shell, gating, spacing, storage)

The page's machinery only. **No module content yet** — that lands in Tasks 7 and 8, which
only append entries to the `COURSE` array. Splitting them keeps this file reviewable.

**Files:**
- Create: `learn/index.html`

**Interfaces:**
- Produces, for Tasks 7 and 8: a global `const COURSE = []` array. Each entry is a **module object**:
  ```
  { id: "m1", week: 1, title: string, blurb: string,
    toy: { file: string, blanks: number, runs: [string, ...] },
    pretest:   [ {id, q, a, why}, ... ],   // gated: answer sealed until a guess is submitted
    concept:   [ htmlString, ... ],        // prose paragraphs. NO CODE.
    falsify:   { run: string, expect: string },
    trace:     [ {file: string, what: string}, ... ],
    payoff:    { p: "P40", title: string, body: htmlString },
    selfcheck: [ {id, q, a, why}, ... ]    // become the NEXT module's retrieval quiz
  }
  ```
  Question `id` values must be **globally unique** (`m1q1`, `m1sc3`, …) — the spaced-review
  queue keys on them.
- Produces these functions (Tasks 7/8 need none of them; they only push data):
  `renderCourse()`, `askQuestion(q, opts)`, `retrievalFor(moduleIndex)`, `saveGuess(id, text, correct)`.

- [ ] **Step 1: Write `learn/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VisualLLm — Fundamentals</title>
<style>
  :root{
    --bg:#0d1117;--panel:#161b22;--line:#30363d;--ink:#e6edf3;--dim:#8b949e;
    --acc:#58a6ff;--good:#3fb950;--warn:#d29922;--bad:#f85149;
    --mono:ui-monospace,'Cascadia Mono','Roboto Mono',Menlo,Consolas,monospace;
    --sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.65}
  .wrap{max-width:820px;margin:0 auto;padding:0 24px 120px}
  header{padding:56px 0 28px;border-bottom:1px solid var(--line);margin-bottom:32px}
  h1{margin:0 0 6px;font-size:26px;letter-spacing:-.02em}
  header p{margin:0;color:var(--dim);max-width:62ch}
  .bar{display:flex;gap:4px;margin-top:22px}
  .bar i{flex:1;height:4px;border-radius:2px;background:#21262d}
  .bar i.on{background:var(--good)}
  .barlbl{font-size:12px;color:var(--dim);margin-top:8px}

  .mod{border:1px solid var(--line);border-radius:12px;background:var(--panel);margin-bottom:28px;overflow:hidden}
  .mod>h2{margin:0;padding:18px 22px;font-size:17px;border-bottom:1px solid var(--line);background:#1c2128}
  .mod>h2 small{display:block;color:var(--dim);font-size:12px;font-weight:400;margin-top:3px;letter-spacing:.04em}
  .sec{padding:22px;border-bottom:1px solid var(--line)}
  .sec:last-child{border-bottom:0}
  .sec>h3{margin:0 0 4px;font-size:11px;letter-spacing:.11em;color:var(--dim);text-transform:uppercase}
  .sec>.hint{margin:0 0 16px;font-size:13px;color:var(--dim);font-style:italic}
  .sec p{margin:0 0 12px}
  .sec p:last-child{margin-bottom:0}
  code{font-family:var(--mono);font-size:12.5px;background:#0b0f14;border:1px solid var(--line);
    padding:1px 6px;border-radius:5px;color:#a5d6ff}
  .cmd{display:block;font-family:var(--mono);font-size:13px;background:#010409;border:1px solid var(--line);
    border-radius:6px;padding:11px 14px;margin:10px 0;color:#7ee787;overflow-x:auto;white-space:pre}

  .q{background:#0b0f14;border:1px solid var(--line);border-radius:8px;padding:16px;margin-bottom:12px}
  .q .txt{margin-bottom:12px}
  .q input{width:100%;padding:9px 12px;background:#010409;border:1px solid var(--line);border-radius:6px;
    color:var(--ink);font-family:var(--mono);font-size:13px}
  .q input:focus{outline:none;border-color:var(--acc)}
  .q button{margin-top:10px;padding:8px 15px;background:var(--acc);color:#04121f;border:0;border-radius:6px;
    font-weight:600;font-size:13px;cursor:pointer;font-family:inherit}
  .q button:hover{filter:brightness(1.1)}
  .q button:disabled{opacity:.45;cursor:not-allowed}
  .seal{margin-top:9px;font-size:12px;color:var(--dim);font-style:italic}
  .ans{display:none;margin-top:13px;padding:13px;border-radius:6px;font-size:14px;
    background:rgba(210,153,34,.1);border:1px solid rgba(210,153,34,.35)}
  .ans.show{display:block}
  .ans b{display:block;margin-bottom:5px;color:var(--warn)}
  .ans .you{margin-top:8px;font-family:var(--mono);font-size:12px;color:var(--dim)}
  .locked{filter:blur(6px);pointer-events:none;user-select:none;transition:filter .4s}
  .locked.open{filter:none;pointer-events:auto;user-select:auto}
  .gatemsg{font-size:13px;color:var(--warn);margin-bottom:14px}
  .gatemsg.done{display:none}

  .trace{list-style:none;padding:0;margin:0}
  .trace li{padding:9px 0;border-bottom:1px solid #21262d;font-size:14px}
  .trace li:last-child{border-bottom:0}
  .trace .f{font-family:var(--mono);font-size:12.5px;color:#a5d6ff;display:block}
  .payoff{background:linear-gradient(180deg,#1b1206,#0b0f14);border:1px solid #6b4708;border-radius:8px;padding:16px}
  .payoff h4{margin:0 0 8px;font-size:14px;color:var(--warn)}
  .empty{color:var(--dim);font-size:13px;font-style:italic}
  footer{padding:40px 0;color:var(--dim);font-size:13px;text-align:center}
  .reset{background:none;border:1px solid var(--line);color:var(--dim);padding:6px 12px;
    border-radius:6px;font-size:12px;cursor:pointer;font-family:inherit}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>VisualLLm — the four foundations</h1>
    <p>One conversation turn, followed through the system. Each module teaches the layer the turn
       crosses, then hands you the real bug it explains. Guess before you read — a wrong guess
       beats a read answer, and that is not a metaphor, it is the finding.</p>
    <div class="bar" id="bar"></div>
    <div class="barlbl" id="barlbl">0 questions answered</div>
  </header>

  <main id="course"></main>

  <footer>
    Toys live in <code>learn/*.py</code> — this page never contains code, so the two can't drift.<br>
    Design + evidence: <code>docs/superpowers/specs/2026-07-11-visualllm-study-plan-design.md</code><br><br>
    <button class="reset" id="reset">Erase my progress</button>
  </footer>
</div>

<script>
/* ---------------- storage: an IMMUTABLE guess log ---------------- */
const KEY = "visualllm-learn-v1";
const store = JSON.parse(localStorage.getItem(KEY) || '{"guesses":{},"missed":[]}');
function persist(){ localStorage.setItem(KEY, JSON.stringify(store)); }

function saveGuess(id, text, correct){
  if (store.guesses[id]) return;            // first guess only. never editable.
  store.guesses[id] = { text, correct, at: new Date().toISOString() };
  if (!correct && !store.missed.includes(id)) store.missed.push(id);
  if (correct) store.missed = store.missed.filter(m => m !== id);
  persist(); paintProgress();
}

/* A guess is "correct" if the expected answer appears in it, digits-and-letters only.
   Deliberately loose: this is retrieval practice, not an exam. */
function judge(given, expected){
  const norm = s => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  return norm(given).includes(norm(expected)) && norm(given).length > 0;
}

/* ---------------- one gated question ---------------- */
function askQuestion(q, onAnswered){
  const prev = store.guesses[q.id];
  const el = document.createElement("div");
  el.className = "q";
  el.innerHTML =
    '<div class="txt">' + q.q + '</div>' +
    '<input type="text" placeholder="commit a guess — wrong is fine, guessing is the point">' +
    '<button>Lock in my guess</button>' +
    '<div class="seal">The answer is sealed until you commit.</div>' +
    '<div class="ans"><b></b><span class="why"></span><div class="you"></div></div>';

  const inp = el.querySelector("input"), btn = el.querySelector("button");
  const ans = el.querySelector(".ans"), seal = el.querySelector(".seal");

  function reveal(text, correct){
    ans.querySelector("b").textContent = (correct ? "Right — " : "The answer: ") + q.a;
    ans.querySelector(".why").innerHTML = q.why;
    ans.querySelector(".you").textContent = 'you said: "' + text + '"';
    ans.classList.add("show");
    seal.style.display = "none";
    inp.disabled = btn.disabled = true;
    inp.value = text;
    if (onAnswered) onAnswered();
  }

  btn.addEventListener("click", () => {
    const v = inp.value.trim();
    if (!v) { inp.focus(); return; }
    const ok = judge(v, q.a);
    saveGuess(q.id, v, ok);
    reveal(v, ok);
  });
  inp.addEventListener("keydown", e => { if (e.key === "Enter") btn.click(); });

  if (prev) reveal(prev.text, prev.correct);   // already answered: stays revealed, stays locked
  return el;
}

/* ---------------- spaced interleaved retrieval ---------------- */
/* Module N opens with questions drawn from EARLIER modules — never its own.
   Anything you missed is re-queued first; the rest fills up to 5. */
function retrievalFor(i){
  if (i === 0) return [];
  const pool = [];
  for (let k = 0; k < i; k++)
    COURSE[k].selfcheck.concat(COURSE[k].pretest).forEach(q => pool.push(q));
  const missed = pool.filter(q => store.missed.includes(q.id));
  const rest   = pool.filter(q => !store.missed.includes(q.id));
  return missed.concat(rest).slice(0, 5);
}

/* ---------------- progress: questions answered, never sections scrolled ---------------- */
function allQuestions(){
  return COURSE.flatMap(m => m.pretest.concat(m.selfcheck));
}
function paintProgress(){
  const qs = allQuestions(), done = qs.filter(q => store.guesses[q.id]).length;
  const bar = document.getElementById("bar");
  bar.innerHTML = "";
  COURSE.forEach(m => {
    const mq = m.pretest.concat(m.selfcheck);
    const d = mq.filter(q => store.guesses[q.id]).length;
    const i = document.createElement("i");
    if (mq.length && d === mq.length) i.className = "on";
    bar.appendChild(i);
  });
  const missed = store.missed.length;
  document.getElementById("barlbl").textContent =
    done + " of " + qs.length + " questions answered" +
    (missed ? "  ·  " + missed + " re-queued for review" : "");
}

/* ---------------- render ---------------- */
function section(title, hint){
  const s = document.createElement("section");
  s.className = "sec";
  s.innerHTML = "<h3>" + title + "</h3>" + (hint ? '<p class="hint">' + hint + "</p>" : "");
  return s;
}

function renderCourse(){
  const root = document.getElementById("course");
  root.innerHTML = "";
  if (!COURSE.length) {
    root.innerHTML = '<p class="empty">No modules loaded yet.</p>';
    return;
  }

  COURSE.forEach((m, i) => {
    const mod = document.createElement("article");
    mod.className = "mod";
    mod.innerHTML = "<h2>" + m.title + "<small>WEEK " + m.week + " · " + m.blurb + "</small></h2>";

    /* 1. cold pretest — gates everything below it */
    const pre = section("1 · Cold pretest",
      "Answer before you read anything below. Being wrong here is the mechanism, not a failure.");
    const gated = [];
    let answered = 0;
    m.pretest.forEach(q => {
      pre.appendChild(askQuestion(q, () => {
        if (++answered >= m.pretest.length) gated.forEach(g => g.classList.add("open"));
        const gm = mod.querySelector(".gatemsg");
        if (answered >= m.pretest.length && gm) gm.classList.add("done");
      }));
    });
    mod.appendChild(pre);

    /* 2. spaced retrieval from EARLIER modules */
    const rq = retrievalFor(i);
    if (rq.length) {
      const r = section("2 · Spaced retrieval",
        "From earlier weeks. Studied once is not studied — this is the half that makes it stick.");
      rq.forEach(q => r.appendChild(askQuestion(q)));
      mod.appendChild(r);
    }

    /* everything from here is blurred until the pretest is committed */
    const gate = document.createElement("div");
    gate.className = "gatemsg";
    gate.textContent = "↓ Sealed until you commit your pretest guesses above.";
    mod.appendChild(gate);

    const con = section("3 · Concept", null);
    m.concept.forEach(p => { const x = document.createElement("p"); x.innerHTML = p; con.appendChild(x); });
    mod.appendChild(con); gated.push(con);

    const lab = section("4 · The lab",
      "Open the file in your editor. Fill only the TODO(you) blanks. Then run it.");
    lab.innerHTML += "<p>File: <code>" + m.toy.file + "</code> — <b>" + m.toy.blanks +
      "</b> blanks are yours.</p>";
    m.toy.runs.forEach(r => { const c = document.createElement("code"); c.className = "cmd"; c.textContent = r; lab.appendChild(c); });
    mod.appendChild(lab); gated.push(lab);

    const fal = section("5 · Falsify",
      "Predict the result of THIS run before you press enter. The gap is the module.");
    const c = document.createElement("code"); c.className = "cmd"; c.textContent = m.falsify.run;
    fal.appendChild(c);
    const fp = document.createElement("p"); fp.innerHTML = m.falsify.expect; fal.appendChild(fp);
    mod.appendChild(fal); gated.push(fal);

    const tr = section("6 · The same idea, in the real system", null);
    const ul = document.createElement("ul"); ul.className = "trace";
    m.trace.forEach(t => {
      const li = document.createElement("li");
      li.innerHTML = '<span class="f">' + t.file + "</span>" + t.what;
      ul.appendChild(li);
    });
    tr.appendChild(ul); mod.appendChild(tr); gated.push(tr);

    const pay = section("7 · The bug this explains",
      "Read it only now — you should already be able to predict it.");
    const box = document.createElement("div"); box.className = "payoff";
    box.innerHTML = "<h4>" + m.payoff.p + " — " + m.payoff.title + "</h4>" + m.payoff.body;
    pay.appendChild(box); mod.appendChild(pay); gated.push(pay);

    const sc = section("8 · Self-check",
      "These come back to hunt you at the start of next week. Miss one and it repeats sooner.");
    m.selfcheck.forEach(q => sc.appendChild(askQuestion(q)));
    mod.appendChild(sc); gated.push(sc);

    /* if the pretest was already committed in a past session, open the gate immediately */
    if (m.pretest.every(q => store.guesses[q.id])) {
      gated.forEach(g => g.classList.add("open"));
      gate.classList.add("done");
    } else {
      gated.forEach(g => g.classList.add("locked"));
    }

    root.appendChild(mod);
  });
  paintProgress();
}

document.getElementById("reset").addEventListener("click", () => {
  if (!confirm("Erase every guess and start over?")) return;
  localStorage.removeItem(KEY);
  location.reload();
});

/* ================= MODULE CONTENT =================
   Tasks 7 and 8 push module objects into COURSE. Nothing else in this file changes.
   HARD RULE: no Python in this file, ever. Name the file and the blank; never the code. */
const COURSE = [];

renderCourse();
</script>
</body>
</html>
```

- [ ] **Step 2: Verify the anti-drift rule holds (the one real invariant)**

Run: `grep -nE "^\s*(def |import |from .* import )|asyncio\.|NotImplementedError" learn/index.html`
Expected: **no output.** Any hit means Python leaked into the page — remove it. This grep is the
enforcement mechanism for the anti-drift rule and must be re-run in Tasks 7 and 8.

- [ ] **Step 3: Verify the page loads with zero modules**

Open `learn/index.html` in a browser.
Expected: the header renders, and the body shows "No modules loaded yet." No console errors.

- [ ] **Step 4: Commit**

```bash
git add learn/index.html
git commit -m "feat(learn): course engine -- gated answers, spaced retrieval, immutable guess log"
```

---

### Task 7: Course content — Modules 1 and 2

**Files:**
- Modify: `learn/index.html` (replace the line `const COURSE = [];` with the array below; the rest untouched)

**Interfaces:**
- Consumes: the module-object shape and `renderCourse()` from Task 6.
- Produces: `COURSE[0]` and `COURSE[1]`, whose `selfcheck`/`pretest` ids become Module 2's and
  Module 3's spaced-retrieval pool.

- [ ] **Step 1: Replace `const COURSE = [];` with Modules 1 and 2**

```javascript
const COURSE = [
{
  id: "m1", week: 1, title: "Sound is just numbers",
  blurb: "the mic reaches the machine",
  toy: { file: "learn/m1_vad.py", blanks: 2,
    runs: ["python learn/m1_vad.py output/q_ai.wav"] },
  pretest: [
    { id:"m1q1",
      q:"A 3-second, 16 kHz, mono, <code>int16</code> WAV. How many <b>bytes</b> of audio data?",
      a:"96000",
      why:"16000 samples per second &times; 3 seconds &times; <b>2 bytes per sample</b> = 96,000. Everything in Module 1 is that one multiplication, read in different directions." },
    { id:"m1q2",
      q:"The repo's <code>output/q_ai.wav</code> — what sample rate would you bet on?",
      a:"24000",
      why:"It is <b>24 kHz</b>, not the 16 kHz the pipeline uses. If you guessed 16000, you just made the exact assumption the toy is built to punish: every timestamp you computed would be 1.5&times; wrong. Read the header. Never assume the rate." }
  ],
  concept: [
    "Sound arrives as a list of numbers. Each number is one <b>sample</b> — the air pressure at one instant — and in this system each sample is an <code>int16</code>: a signed 16-bit integer, which is <b>exactly two bytes</b>.",
    "That gives you the only equation you need: <b>bytes = seconds &times; rate &times; 2</b>. Read it forward to size a buffer, backward to turn a byte offset into a timestamp. The avatar server, the TTS client and the VAD all live or die on this one line.",
    "And it gives you the trap. If a stream is two bytes per sample, then a <b>one-byte</b> shift re-pairs every byte with the wrong neighbour. Sample <i>n</i> is now built from the second half of one sample and the first half of the next. The result is still a valid WAV. It still has plausible loudness. It is noise."
  ],
  falsify: {
    run: "python learn/m1_vad.py output/q_ai.wav --drop-byte",
    expect: "Predict first, in one line: what happens to the mean RMS, and what happens to the number of speech segments? Then run it. Most people expect a small glitch. What they get is the loudness <i>rising</i> and the speech structure dissolving — because the audio is no longer audio."
  },
  trace: [
    { file:"local_services/musetalk_video.py :: _align_even",
      what:" — carries the odd byte forward instead of dropping it, on the copy you HEAR." },
    { file:"local_services/musetalk_video.py :: self._srv_carry",
      what:" — does the same for the copy sent to the lip-sync model. This line is the fix for P40." },
    { file:"pipeline/stages/vad.py",
      what:" — Silero VAD: the real version of the toy you just built." }
  ],
  payoff: { p:"P40", title:"the lip-sync model was fed noise for three sessions",
    body:"The TTS hands back <b>odd-byte</b> buffers. The audio going to the browser was protected — so <b>the voice always sounded perfect</b>. But the copy sent to the avatar dropped that byte, so MuseTalk lip-synced against pure noise, and since it reads a Whisper transcription <i>of the waveform</i>, the mouth flapped in a generic wordless pattern that never closed for pauses.<br><br>It survived three sessions of debugging because the tests <b>could not fail</b>: an offline render fed audio captured from WebRTC was fed the <i>repaired</i> copy, so it always looked fine. Read the METROLOGY note under P40 in <code>docs/PROBLEMS-AND-FIXES.md</code> — it is the most valuable page in this repo, and it is really about how to build a test that <i>can</i> fail." },
  selfcheck: [
    { id:"m1sc1", q:"Byte 48,000 of a 16 kHz mono int16 stream. What timestamp is that?", a:"1.5",
      why:"48000 bytes / (16000 &times; 2 bytes per second) = <b>1.5 seconds</b>." },
    { id:"m1sc2", q:"The voice sounds perfect but the mouth flaps wrong. Which copy of the audio is broken?", a:"avatar",
      why:"The one going to the <b>avatar</b>. Two copies leave the TTS; only the downstream (heard) one was protected. A bug in the branch you cannot hear is invisible to your ears — which is exactly why it hid." },
    { id:"m1sc3", q:"Why is correlating mouth motion against audio RMS a useless test?", a:"noise",
      why:"Because <b>noise has RMS too</b>. A mouth flapping at loud garbage correlates beautifully with the loudness of that garbage. P40's note records that this test misled us four separate times." }
  ]
},
{
  id: "m2", week: 2, title: "Streams, queues, and time",
  blurb: "the words become a stream",
  toy: { file: "learn/m2_pipeline.py", blanks: 3,
    runs: ["python learn/m2_pipeline.py unbounded",
           "python learn/m2_pipeline.py bounded",
           "python learn/m2_pipeline.py paced"] },
  pretest: [
    { id:"m2q1",
      q:"A fast producer feeds a slow consumer through a queue with <b>no size limit</b>. Nothing crashes, nothing is dropped. What goes wrong anyway?",
      a:"latency",
      why:"<b>Latency.</b> The queue absorbs the mismatch by getting longer, so by the end you are delivering something made many seconds ago. An unbounded queue does not fix a slow consumer — it <i>hides</i> it, and bills you in staleness." },
    { id:"m2q2",
      q:"The LLM has produced 4 words of a 40-word answer. Why does the TTS start speaking already?",
      a:"stream",
      why:"Because the pipeline <b>streams</b>. The first sentence reaches TTS before the full answer exists, and TTS's first audio chunk reaches the avatar immediately. Time-to-first-output is a property of the first <i>piece</i>, never the whole reply — that is the entire reason this system can answer in under 3 seconds." }
  ],
  concept: [
    "Every stage in <code>pipeline/main.py</code> is a coroutine, and between them sit queues. A stage <code>await</code>s its input, does its work, and <code>await</code>s a put on its output. When one stage blocks, the event loop simply runs another — that is all <code>asyncio</code> is doing.",
    "The queue's <b>size limit</b> is the whole design. Unbounded, a fast producer runs free and the backlog turns into latency. Bounded, <code>await queue.put()</code> <i>blocks the producer</i> the moment the consumer falls behind, forcing the whole pipeline to the speed of its slowest stage. That blocking is called <b>backpressure</b>, and it is a feature.",
    "Then there is <i>when</i> you let something out. Free-run means: ship it the instant it is ready. Paced means: item N leaves at <code>t0 + N/fps</code>, on a clock, no matter when it arrived. Hold that second idea — in Module 4 it turns out to be the system's central design decision."
  ],
  falsify: {
    run: "python learn/m2_pipeline.py unbounded    (then: bounded)",
    expect: "Predict the <b>worst item age</b> in each mode before running — is bounded higher or lower than unbounded, and by roughly how much? Most people expect bounding to make things <i>slower</i> (you're throttling the producer, after all). It makes them dramatically fresher."
  },
  trace: [
    { file:"pipeline/main.py", what:" — the Pipeline assembly: the stages, in order, and the frames that flow between them." },
    { file:"local_services/first_piece_aggregator.py", what:" — splits off a short opening clause so speech can start before the sentence is finished. Pure streaming latency trick." },
    { file:"pipeline/metrics.py :: TtfoMeter", what:" — measures UserStoppedSpeaking to BotStartedSpeaking. Note carefully WHERE it stops measuring." }
  ],
  payoff: { p:"P35", title:"our own latency metric was lying by 1.26 seconds",
    body:"<code>[TTFO]</code> measured up to <code>BotStartedSpeakingFrame</code> — a frame emitted <b>inside the server</b>. But the user's ear is at the far end of a transport, a WebRTC encode, a network, and a browser jitter buffer. All of that was <b>after</b> the stopwatch stopped, and it was worth ~1.26 seconds.<br><br>We were tuning hard against a number that did not describe the experience we were selling. The fix (<code>scripts/measure.py</code>) stitches the client's arrival clock onto the server's <code>t0</code> and reports a full waterfall <i>to the ear</i>. The lesson is bigger than the bug: <b>a metric that stops before the user does is not a metric, it is a comfort.</b>" },
  selfcheck: [
    { id:"m2sc1", q:"One word: what does a bounded queue do to a producer that is outrunning its consumer?", a:"block",
      why:"It <b>blocks</b> it (<code>await put()</code> does not return). That is backpressure — the pipeline self-regulates to its slowest stage instead of building a hidden backlog." },
    { id:"m2sc2", q:"Time-to-first-output depends on the length of the whole reply. True or false?", a:"false",
      why:"<b>False</b> — and this is the point of streaming. TTFO depends on the first <i>piece</i>. Hence <code>COSYVOICE_FIRST_PIECE</code>: emit a short opening clause, and first audio lands ~1.3s sooner even though the full answer takes exactly as long." },
    { id:"m2sc3", q:"Why can a metric that ends at <code>BotStartedSpeakingFrame</code> be honest and still be wrong?", a:"ear",
      why:"Because the frame is emitted in the server, and the <b>ear</b> is past the transport, the encode, the network and the jitter buffer. Every number it reports is true; the thing it measures is just not the thing that matters. (P35.)" }
  ]
}
];
```

- [ ] **Step 2: Re-run the anti-drift grep**

Run: `grep -nE "^\s*(def |import |from .* import )|asyncio\.|NotImplementedError" learn/index.html`
Expected: **no output.** (The strings `python learn/m2_pipeline.py ...` are shell commands, not Python — they are allowed. `await queue.put()` appears only as prose inside `<code>`, which is fine; if the grep ever flags it, the rule is about *code blocks*, not the word.)

- [ ] **Step 3: Verify in the browser**

Open `learn/index.html`.
Expected: two modules render. Sections 3-8 of Module 1 are **blurred**. Answer both pretest
questions and they un-blur. Module 2 shows a **"Spaced retrieval"** section containing questions
from Module 1. Reload the page: your answers are still there, still locked, still revealed.

- [ ] **Step 4: Commit**

```bash
git add learn/index.html
git commit -m "feat(learn): modules 1-2 -- audio/PCM (P40) and async streaming (P35)"
```

---

### Task 8: Course content — Modules 3, 4, and the capstone

**Files:**
- Modify: `learn/index.html` (append two module objects to `COURSE`, and add the capstone block)

**Interfaces:**
- Consumes: the module shape from Task 6; `COURSE` already holding Modules 1-2 from Task 7.
- Produces: the complete four-module course.

- [ ] **Step 1: Append Modules 3 and 4 to the `COURSE` array**

Insert before the closing `];` of `COURSE`:

```javascript
,
{
  id: "m3", week: 3, title: "Models that generate one step at a time",
  blurb: "the machine finds its words, and its voice",
  toy: { file: "learn/m3_sampler.py", blanks: 4,
    runs: ["python learn/m3_sampler.py prefill",
           "python learn/m3_sampler.py greedy",
           "python learn/m3_sampler.py topp",
           "python learn/m3_sampler.py ras"] },
  pretest: [
    { id:"m3q1",
      q:"Two prompts, one short and one long. Both produce a 5-word answer. Why does the long one take longer to produce its <b>first</b> word?",
      a:"prefill",
      why:"<b>Prefill.</b> Before a model can emit token #1 it must process every token of the input. So time-to-first-token scales with the <i>input</i> length, even when the output is identical. This single fact is why <code>COSYVOICE_FIRST_PIECE</code> exists." },
    { id:"m3q2",
      q:"A TTS model gets stuck emitting the same token over and over — a 4-second sentence becomes 12 seconds of silence. What kind of failure is that?",
      a:"repetition",
      why:"A <b>repetition</b> loop. An autoregressive model feeds its own output back as input, so a self-reinforcing state can trap it. The defence is a sampler that penalises recent tokens — and P18 is what happened when we lost ours." }
  ],
  concept: [
    "An autoregressive model emits <b>one token, feeds it back, and emits the next</b>. Everything strange about its latency and its failures falls out of that loop. Your toy is a character-level Markov chain, not a transformer — but it is autoregressive, and that is the only property that matters here.",
    "Latency splits in two. <b>Prefill</b> processes the input before a single token comes out; <b>decode</b> emits tokens one at a time after. Prefill is why a long sentence handed to CosyVoice costs ~3.0s to first audio while a short clause costs ~1.7s — and why we split the first clause off and speak it early.",
    "Failure also falls out of the loop. Take the likeliest token every time (<b>greedy</b>) and the model can enter a cycle it can never leave. <b>Top-p</b> samples from the smallest set of tokens covering p of the probability mass, which usually escapes. A <b>repetition-aware</b> sampler down-weights tokens it just emitted, which escapes <i>by construction</i>. Guess which one CosyVoice needs."
  ],
  falsify: {
    run: "python learn/m3_sampler.py greedy    (then: ras)",
    expect: "Predict what greedy's output looks like after ~50 characters. Then look at it. The model is not broken and the corpus is not broken — the <i>decoding rule</i> is. Now run <code>ras</code> and watch one line of sampling policy dissolve the whole failure."
  },
  trace: [
    { file:"CosyVoice/cosyvoice/vllm/ras_logits_processor.py  (cosyvoice repo)",
      what:" — the real repetition-aware sampler, restored as a vLLM logits processor. This IS the P18 fix." },
    { file:"local_services/cosyvoice_tts.py",
      what:" — the streaming client: first audio chunk forwarded the moment it exists." },
    { file:"pipeline/stages/llm.py",
      what:" — the OpenRouter/Groq pin. The LLM hop was once the single dominant cost in the whole turn." }
  ],
  payoff: { p:"P18", title:"Chinese looped on silence, and the avatar lip-synced through it",
    body:"Moving CosyVoice's LLM onto vLLM bought a huge speedup — and <b>silently dropped its repetition-aware sampling (RAS)</b>. Chinese then intermittently looped on the silence token: a 4-second sentence became ~12 seconds of dead air. It was heard as 'halting' speech, and the avatar kept lip-moving through the silence, because the mouth follows the waveform.<br><br>The fix restores RAS as a vLLM logits processor. Note the shape of this bug — <b>an optimisation quietly removed a correctness guarantee nobody had written down.</b> That shape recurs: see P33, where CUDA graphs made the stopwatch faster and the <i>lipsync worse</i>, and the eye overruled the benchmark." },
  selfcheck: [
    { id:"m3sc1", q:"Which scales time-to-first-token: the length of the input, or of the output?", a:"input",
      why:"The <b>input</b> — that is prefill. The output length costs you nothing until decoding starts, which is why splitting off a short first clause wins TTFO for free." },
    { id:"m3sc2", q:"One word: what defence stops an autoregressive model from looping forever?", a:"repetition",
      why:"A <b>repetition</b>-aware sampler (RAS): down-weight what you just emitted. Top-p escapes a loop probabilistically; RAS escapes it by construction." },
    { id:"m3sc3", q:"An optimisation makes a benchmark faster but a human says it got worse. Who wins?", a:"human",
      why:"The <b>human</b>. P33 is exactly this: CUDA graphs won the TTS stopwatch and lost the Chinese lipsync, because the graph decode perturbed the RAS sampling and the mouth follows the waveform. The benchmark was measuring the wrong side. The avatar is the product." }
  ]
},
{
  id: "m4", week: 4, title: "Pixels, GPUs, and the sync contract",
  blurb: "a face has to say it in time",
  toy: { file: "learn/m4_sync.py", blanks: 4,
    runs: ["python learn/m4_sync.py live",
           "python learn/m4_sync.py steady",
           "python learn/m4_sync.py steady --contention"] },
  pretest: [
    { id:"m4q1",
      q:"The avatar should render 14 frames a second. The GPU is busy running the TTS, so it manages 10. The voice plays at exactly one second per second. After a 20-second reply, how far behind are the lips?",
      a:"seconds",
      why:"<b>Seconds</b> behind — and, crucially, the gap <i>grows with the length of the turn</i>. Every second you lose 4 frames, and nothing ever gives them back. A short reply looks fine; a long one falls apart. That is why the bug (P16) only showed up on long answers." },
    { id:"m4q2",
      q:"You cannot make the renderer faster. Do you let the video fall behind the voice, or make the voice wait for the video?",
      a:"wait",
      why:"This system makes the voice <b>wait</b> (<code>MUSETALK_SYNC_MODE=steady</code>): audio is released paced to the frames actually rendered, so drift is zero by construction and the cost becomes a brief pause. The alternative (<code>live</code>) never pauses but the lips trail. There is no third option — and that is the whole point of the module." }
  ],
  concept: [
    "One 16 GB card runs both the TTS and the avatar. They contend, so the renderer's real frame rate is a <i>variable</i>, not a constant — while audio's rate is fixed by physics. That mismatch has to be paid somewhere, and choosing <b>where</b> is the design decision.",
    "<b>Audio-master</b> (<code>live</code>): ship the voice immediately, let video arrive when it can. Nothing ever pauses; the lips slide behind, and the drift accumulates with turn length. <b>Video-master</b> (<code>steady</code>): pin audio frame N to video frame N and release both together. Drift becomes structurally impossible; a render stall becomes a brief silence instead.",
    "The user chose <code>steady</code>: a pause is survivable, lips that slide off the words are not. Everything else in this layer — TensorRT, the GPU composite, <code>MUSETALK_LEAD_FRAMES</code> — exists to make that choice cheap by keeping the real frame rate above the target."
  ],
  falsify: {
    run: "python learn/m4_sync.py live    (then: steady --contention)",
    expect: "Predict <code>live</code>'s final drift for an 8-second turn, and predict what <code>steady</code> does with that same lost time — it cannot vanish. Watch where each mode <b>puts the pain</b>: one spends it on your eyes, the other on your ears. The engineering is choosing which."
  },
  trace: [
    { file:"local_services/musetalk_server/app.py", what:" — the render loop, and video_clock: it counts only REAL rendered frames, never held duplicates." },
    { file:"local_services/musetalk_video.py", what:" — the client: steady vs live, and the paced release queue. Your toy, for real." },
    { file:"pipeline/main.py :: video_out_is_live = not config.avatar_sync_with_audio",
      what:" — the CRITICAL COUPLING. Pipecat only honours per-frame A/V pinning when the transport is NOT live; set is_live independently and the tagged frames are silently dropped and video free-runs." }
  ],
  payoff: { p:"P16", title:"the lips drifted further behind the longer the reply got",
    body:"Under shared-GPU contention the PyTorch render path fell below the target frame rate. Drift is <code>turn_length &times; (1 - real_fps/target_fps)</code> — so it <b>scaled with the reply</b>, invisible on a short answer and ruinous on a long one. The fix was TensorRT (<code>MUSETALK_TRT=1</code>, render ~389ms &rarr; ~255ms per segment), which keeps the real rate above the target so <code>steady</code> rarely has to pause at all.<br><br>Note what the fix actually did: it did not repair the sync logic, which was correct all along. It bought back the <i>headroom</i> the sync logic needed. Also read <b>P1</b> — <code>cudnn.benchmark = True</code> re-autotuned on each turn's first segment and cost a <b>16-second</b> GPU spike, every single turn." },
  selfcheck: [
    { id:"m4sc1", q:"Why does A/V drift grow with reply length instead of staying constant?", a:"accumulate",
      why:"Because the frame deficit <b>accumulates</b>: every second at 10fps against a 14fps target loses 4 frames, permanently. Drift = turn_length &times; (1 - real/target). Short replies hide it completely." },
    { id:"m4sc2", q:"In <code>steady</code> mode the renderer stalls. What does the user experience?", a:"pause",
      why:"A brief <b>pause</b> in the voice — then it resumes in sync. The lost time has to go somewhere; steady spends it on the ears rather than on the eyes." },
    { id:"m4sc3", q:"Per-frame A/V pinning silently stops working. What is the first thing you check?", a:"is_live",
      why:"<code>video_out_is_live</code>. Pipecat 1.3.0 only reads the tagged frames when the transport is <b>not</b> live; with <code>is_live=True</code> they are dropped without a word and video free-runs. Hence the hard coupling in <code>main.py</code>." }
  ]
}
];
```

- [ ] **Step 2: Add the capstone, immediately after `renderCourse();`**

```javascript
/* ---------------- capstone ---------------- */
const cap = document.createElement("article");
cap.className = "mod";
cap.innerHTML =
  '<h2>Capstone — one real turn, all the way down<small>AFTER WEEK 4 · ~2 hours</small></h2>' +
  '<section class="sec"><h3>Drive the real system</h3>' +
  '<code class="cmd">python -m scripts.measure --offline-capture</code>' +
  '<p>Then write the waterfall yourself. For <b>every hop</b> of one turn, name four things: ' +
  'the <b>layer</b>, the <b>file</b>, the <b>milliseconds</b>, and the <b>one knob</b> that moves it. ' +
  'If you can fill that table without looking anything up, the four modules did their job.</p></section>' +
  '<section class="sec"><h3>The defence</h3>' +
  '<p>Open <code>docs/PROBLEMS-AND-FIXES.md</code>, pick a P-number you have never read, and explain ' +
  'it <b>before</b> you read the write-up. Not the fix — the <i>mechanism</i>. You should now be able to ' +
  'get most of the way there from first principles, because every one of them is one of these four ' +
  'layers, misbehaving.</p></section>' +
  '<section class="sec"><h3>Then: weeks 6 and 10</h3>' +
  '<p>Come back and re-answer your missed questions. This is not optional decoration — <b>spacing is ' +
  'half of why any of this will still be in your head next term.</b> Studied once is not studied. ' +
  'Anything that decays before week 10 was activity, not learning.</p></section>';
document.getElementById("course").appendChild(cap);
```

- [ ] **Step 3: Re-run the anti-drift grep**

Run: `grep -nE "^\s*(def |import |from .* import )|asyncio\.|NotImplementedError" learn/index.html`
Expected: **no output.**

- [ ] **Step 4: Verify the whole course in the browser**

Open `learn/index.html`. Check all of:
- Four modules plus the capstone render.
- Module 1's sections 3-8 are blurred until both pretest questions are committed.
- Module 4's "Spaced retrieval" section draws questions from Modules 1-3.
- Deliberately answer one question **wrong**; the progress line reports it as re-queued, and it
  appears first in the next module's retrieval section.
- Reload: every guess persists, still locked, still revealed.
- "Erase my progress" clears it and the gates close again.

- [ ] **Step 5: Commit**

```bash
git add learn/index.html
git commit -m "feat(learn): modules 3-4 + capstone -- inference (P18) and A/V sync (P16)"
```

---

### Task 9: Wire `learn/` into the repo's docs

**Files:**
- Modify: `README.md` (add one line to the repo's doc list)

**Interfaces:**
- Consumes: the finished `learn/` folder.
- Produces: discoverability. Nothing depends on this.

- [ ] **Step 1: Find the docs list in `README.md`**

Run: `grep -n "STATUS.md\|WORKFLOW.md" README.md`
Expected: the lines listing the repo's key docs. Add the new entry next to them, matching the
surrounding style exactly (do not restructure the list).

- [ ] **Step 2: Add the line**

```markdown
- **`learn/`** — the fundamentals course. Open `learn/index.html`: four modules that follow one turn
  through the system (audio, streaming, inference, GPU/sync), each ending on the real bug it explains.
```

- [ ] **Step 3: Verify nothing else changed**

Run: `git diff --stat README.md`
Expected: `1 file changed, 2 insertions(+)` — and nothing else.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: point README at the learn/ fundamentals course"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: the four modules → Tasks 2-5 (toys) and 7-8
(teaching); the eight-step module structure → the `renderCourse()` sections in Task 6; gated
pretest, spaced interleaved retrieval, honest progress and the immutable guess log → Task 6's
engine; faded blanks at 20/40/55/70% → Tasks 2/3/4/5 (2, 3, 4, 4 blanks respectively, over toys of
rising conceptual difficulty); the anti-drift rule → the grep in Tasks 6, 7 and 8; the week-6/week-10
checkpoints → the capstone block in Task 8; `learn/` as a root-level sibling folder → Task 1.

**Deliberate deviation from the skill's default:** there is **no TDD cycle**, because `CLAUDE.md`
forbids inventing a test suite in this repo. Its role is taken by two real verifications: *run the
toy and observe the stated output* (each toy's Step 3 names the specific observable that must hold,
and says to fix the toy if it does not), and *the anti-drift grep* (a genuine invariant that can
fail). Toy Steps 3-4 also guarantee the blanks ship **unfilled** — a toy that arrives complete has
silently destroyed the entire pedagogy, so it is checked with `grep -c NotImplementedError`.

**Placeholder scan:** none. Every question, answer, explanation, concept paragraph, trace entry and
bug write-up is complete in the plan; no task says "similar to" another.

**Type consistency:** the module object shape declared in Task 6's Interfaces (`id`, `week`, `title`,
`blurb`, `toy{file,blanks,runs}`, `pretest[]`, `concept[]`, `falsify{run,expect}`, `trace[]`,
`payoff{p,title,body}`, `selfcheck[]`) is used identically in Tasks 7 and 8. Question ids are
globally unique (`m1q1`, `m1sc1`, … `m4sc3`) as the retrieval queue requires. Filenames match
between the toys (Tasks 2-5) and the `toy.file` strings (Tasks 7-8).
