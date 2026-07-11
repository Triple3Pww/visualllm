# Module 3 — Models that generate one step at a time

## Who I am and what I'm doing

I'm Chanachon, a 3rd-year ICT student at Mahidol, on a research internship. I built a real-time system
where you speak into a browser and a photoreal talking-head avatar answers you out loud: microphone →
detect speech → speech-to-text → LLM → text-to-speech → face-animation model → back to the browser, all
on one GPU. I built it with heavy AI help and I'm learning the fundamentals underneath, one layer per
week. So far: audio is int16 and corruption is loud (Module 1); the event loop, bounded queues, and
streaming (Module 2).

**This module is about the two AI models in my pipeline — the LLM and the text-to-speech — and how they
actually generate output.** This is the hardest week for me because I have no deep-learning background,
so please go slow and skip the math. I only need how these models *behave*, not how they're trained. I
have a small Python exercise after, so explain ideas, don't write code for me. Take two sessions if it
needs it.

**Out of scope — please don't teach me:** training, backpropagation, loss functions, the attention math,
or tokenizer internals. Just inference behaviour.

---

## What I need to learn

### What a model is (at inference)
A function with frozen, learned parameters. Given some context, it outputs a **score for every possible
next token**. That's it. No learning happens while it runs. A language model is a *next-token scorer*;
chat, agents, everything else is a loop wrapped around a next-token scorer.

### Tokens
Text is chopped into **tokens** — roughly, common words and word-fragments — from a fixed vocabulary of
maybe 30k–100k. The model emits **one token at a time**, not one word or sentence.

### Autoregression (the core idea)
```
context = the prompt
loop:
    scores = model(context)      # a score for every token
    token  = pick_one(scores)    # the sampling step (below)
    output += token
    context = context + token    # ← its own output becomes its next input
```
That last line is everything. The model feeds its own output back in as input. If it makes a mistake at
step 5, at step 6 it sees that mistake as if it were fact — there's no undo. Every strange behaviour
below comes from this loop.

### Prefill vs decode (why the *input* length costs you)
Before the model can emit token #1, it must process **every token of the input** — that's **prefill**.
Emitting tokens one at a time afterward is **decode**. Consequence, and it's counterintuitive:
**time-to-first-token scales with the INPUT length, not the output.** A long prompt gives a late first
token even if the answer is one word. In my exercise, prompts of 20 / 200 / 1500 characters give
first-token times of **0.016 / 0.109 / 0.813 s** — before any output exists at all. It's linear in input
length.

This is why my system splits off a short opening clause: handing my TTS a whole long sentence costs ~3.0 s
before the first sound; a short clause costs ~1.7 s. Nothing got faster — the work was reordered so the
part I'm waiting on is cheap. (That's the same idea as Module 2's streaming, one layer down.)

### Sampling — picking the token
The model gives a score for every token; softmax turns those into probabilities. Which one do you pick?
- **Greedy**: always the highest. Deterministic.
- **Temperature**: flatten or sharpen the distribution first.
- **Top-k**: sample among the k likeliest.
- **Top-p (nucleus)**: sample among the smallest set of tokens whose probabilities sum to p (e.g. 0.9) —
  the set shrinks when the model is confident, grows when it isn't. The modern default. (Better than
  top-k because top-k injects randomness even when the model is 99% sure.)
- **Repetition penalty**: down-weight tokens that appeared recently.

### The punchline — repetition loops
Greedy sounds best (always the most likely token) but it can enter a cycle it never escapes: it picks A,
which makes B likeliest, which makes A likeliest again — forever. Because the output is fed back as input
(autoregression), a self-reinforcing state is a trap with no built-in escape. In my exercise, greedy locks
into `(`tailscale (`tailscale (`tailscale…` and never stops. Top-p usually escapes but only
*probabilistically*. A **repetition-aware sampler** (down-weight recently-used tokens) escapes *by
construction*. There's even a tuning subtlety: with too short a look-back window the tight loop dies but a
*longer* one forms outside the window; too aggressive and you build new loops. Sampling policy is a
**correctness** surface, not a style knob.

### How a neural TTS works
```
text → autoregressive model → audio tokens → vocoder → waveform (int16 samples — back to Module 1)
```
A text-to-speech model emits audio tokens exactly the way an LLM emits word tokens — **at its core it *is*
an autoregressive model**, so everything above (prefill cost, repetition loops, sampling) applies to it
too. Also: **zero-shot voice cloning** means you give it a few seconds of reference audio and it speaks in
that voice with no retraining — that's how my avatar has a specific person's voice.

### What vLLM is (and why it's risky)
**vLLM** is a fast inference server (I run my TTS on it). It's fundamentally a *throughput* optimisation,
and my product is a *latency* product with one user — so it helped, but for reasons I should be able to
explain. The important part: **vLLM is a different *implementation* of the same model.** Hold that — it's
the bug below.

---

## The bug this explains ("P18" in my project)

I moved my TTS's internal language model onto vLLM for speed. Everything benchmarked better. Then Chinese
synthesis started intermittently producing **12 seconds of dead silence** where a 4-second sentence should
be — heard as "halting" speech — and the avatar kept moving its mouth through the silence.

The cause: vLLM's sampling pipeline **silently dropped the repetition-aware sampler** the original
implementation had. Without it, the model looped on the silence token — exactly the degenerate cycle from
my exercise. And because the mouth follows the waveform (Module 1), the avatar lip-synced through nothing.
The fix was to reimplement the repetition-aware sampler for vLLM. The general shape: **an optimisation
silently removed a correctness guarantee nobody had written down.** It happened again later in the same
project — turning on a GPU optimisation ("CUDA graphs") made benchmarks faster but the lip-sync *worse*,
because it perturbed the same sampling. The stopwatch said better, my eye said worse, and my eye was right.

---

## Questions for Gemini to ask me

1. What does a language model actually output at each step?
2. In the autoregressive loop, what does the model see at step 6 if it made a mistake at step 5?
3. Which scales time-to-first-token — the input length or the output length? Why?
4. Why does my system split off a short opening clause before synthesising?
5. Why can greedy decoding loop forever, and what stops it?
6. What is top-p sampling, in one sentence? Why is it better than top-k when the model is confident?
7. Why is a repetition-aware sampler better than top-p at killing loops?
8. In what sense is a text-to-speech model "the same machine" as a language model?
9. I swap in a faster inference engine and all benchmarks improve. What could have silently broken?
10. A benchmark says a change made things faster; a human says the output got worse. Who wins?

If I miss 3, 5, or 9, make me work through it again — those are the heart of this module.
