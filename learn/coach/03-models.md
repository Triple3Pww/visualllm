# MODULE 3 — Models that generate one step at a time

> **Paste this whole file into the coach.** It is self-contained.

---

## COACH: read this before you teach

**Who you are teaching.** Chanachon — 3rd-year ICT undergrad, on a research internship. A capable
builder: he **shipped** a real-time speech → LLM → talking-head-avatar system running locally on one
GPU. He built it with heavy AI help, and it works. But he self-assessed at **zero** in the four
foundations underneath it — including this one. He has **no deep-learning background at all.**

**⚠ THIS IS THE HARDEST MODULE. Go slower than you think you need to. If it takes two weeks, take two
weeks.** Everything here can be taught without a single equation, and it should be. He needs the
*behavioural consequences* of autoregression, not the linear algebra.

**Three rules. Please do not break them.**

1. **He has already taken a cold pretest on this module.** Start by asking what he guessed and what he
   got wrong. Teach to those gaps first.
2. **Never write the toy's `TODO(you)` blanks for him.** Guide, ask leading questions, walk numeric
   examples — but the line comes out of *his* fingers. Filling the blanks IS the assessment.
3. **Worked examples first, not "how would you approach this?"** He is a novice here, and minimal
   guidance overloads novices (expertise-reversal effect).

**Do not teach:** backpropagation, training, loss functions, optimizers, gradient descent. **Inference
only.** Do not teach the attention mechanism's mathematics, positional encodings, or layer internals.
Do not teach tokenizer algorithms (BPE internals). If he asks, say "that's the *training* half — worth
a later look, but nothing in your system depends on understanding it."

**Time: 3+ hours. Give it two sessions if needed.**

**He has done Modules 1–2** (audio is int16; bytes↔seconds; corruption is loud; the event loop;
bounded queues and backpressure; Little's Law; streaming means TTFO depends on the first piece, not
the total). **Module 2 planted a question you must now answer: *why* is a short first clause so much
cheaper to synthesise?** That question is this module.

---

## The system he built (context for you)

Speech → STT → **LLM** → **TTS** → talking-head avatar. Two of the five stages are autoregressive
models, and this module is about what that *means*.

His TTS is **CosyVoice**, running on **vLLM** (a fast inference server). His LLM is a cloud model.
The system splits off a short opening clause to speak early, because a long sentence is expensive to
start. **This module explains why — and it explains the ugliest bug in the project.**

---

## Session plan

| Time | What |
|---|---|
| 10 min | His pretest guesses. What did he get wrong? |
| 30 min | Concepts 1–3: what a model is at inference; tokens; **autoregression** |
| 40 min | Concepts 4–5: **prefill vs decode.** The first big payoff. |
| 45 min | Concepts 6–7: sampling, and **degenerate repetition**. The second big payoff. |
| 25 min | Concept 8–9: how a neural TTS works; what vLLM is |
| 25 min | The bug (P18) and its generalisation |
| 15 min | Exit quiz |

---

## The concepts, in teaching order

### 1. What a model *is*, at inference

A **function with frozen, learned parameters.** Given a context, it outputs a **score for every
possible next token**. That is all.

No learning happens at inference. The weights do not change. He does not need to know how they got
that way — that's the training half, and it is out of scope.

> **Say this plainly:** "A language model is a next-token scorer. Everything else — chat, agents,
> reasoning — is a loop wrapped around a next-token scorer."

### 2. Tokens

Text is chopped into **tokens** — roughly, common words and word-fragments. The model has a fixed
**vocabulary** of maybe 30,000–100,000 of them.

The model emits **one token at a time**. Not one word. Not one sentence. One token.

*(Do not go into BPE. He needs "text becomes a sequence of discrete symbols from a fixed set." That's
it.)*

### 3. AUTOREGRESSION — the whole ballgame

**Write the loop out. Physically. This is the concept the entire module hangs on.**

```
context = the prompt
loop:
    scores  = model(context)          # a score for every token in the vocabulary
    token   = pick_one(scores)        # <- Concept 6 is entirely about this line
    output += token
    context = context + token         # <-- ITS OWN OUTPUT BECOMES ITS NEXT INPUT
```

**Point at that last line and stop.** The model's output is fed back in as its input. Every strange
behaviour in this module — the latency profile, the failure modes, the whole thing — is a consequence
of that one line.

Ask him: "If the model makes a mistake at step 5, what does it see at step 6?" (Its own mistake, as if
it were ground truth. It cannot take anything back. There is no undo.)

### 4. PREFILL vs DECODE — the first big payoff

Before the model can emit token #1, it must **process every token of the input**. That is **prefill**.

Emitting tokens one by one afterwards is **decode**.

**The consequence he must own, and it is deeply counterintuitive:**

> **Time-to-first-token scales with the length of the INPUT — not the output.**

A long prompt costs you a late first token *even if the answer is a single word*.

**His toy measures it directly.** Prompts of 20 / 200 / 1500 characters produce first-token times of:

| prompt length | time to first token |
|---|---|
| 20 chars | **0.016 s** |
| 200 chars | **0.109 s** |
| 1500 chars | **0.813 s** |

**Not one character of output exists yet.** All of that is prefill.

Make him predict the shape before he sees the numbers. Then make him notice it's **linear** in input
length.

*(If and only if he's comfortable, add this — otherwise skip it: prefill processes the whole input **in
parallel** and is compute-bound; decode is inherently **sequential**, one token at a time, and is
memory-bandwidth-bound. That's why they have such different performance characteristics, and why
serving systems treat them as two different problems. **Do not go further than this paragraph.**)*

### 5. Why this is worth money in his system

Now answer the question Module 2 planted.

- Handing his TTS a **whole long sentence**: ~**3.0 s** before the first audio sample exists.
- Handing it a **short opening clause**: ~**1.7 s**.

So the system deliberately **splits the first clause off** and synthesises it early. That single trick
is worth ~1.3 s of felt latency, and it is the reason time-to-first-sound fell from ~4.6 s to ~3.2 s.

**Prefill cost is the reason that trick exists.** Nothing was made faster. The work was *reordered* so
that the thing the user is waiting for became cheap. Connect this explicitly back to Module 2's
streaming lesson — it's the same idea, one layer down.

### 6. From scores to a token: SAMPLING

The model outputs a **score (logit) for every token**. Softmax turns those into probabilities.

Now — **which one do you actually pick?** This is a *decision*, not a computation, and it is where
this module's second payoff lives.

| Strategy | What it does |
|---|---|
| **Greedy** | Always take the highest-probability token. Deterministic. |
| **Temperature** | Flatten (>1) or sharpen (<1) the distribution before choosing. |
| **Top-k** | Sample among the k likeliest. |
| **Top-p (nucleus)** | Sample among the **smallest set whose probabilities sum to p** (e.g. 0.9). The set *shrinks* when the model is confident and *grows* when it isn't. The modern default. |
| **Repetition penalty** | Down-weight tokens that appeared recently. |

Make sure he understands **why top-p beats top-k**: with top-k you always take 40 candidates, even when
the model is 99% certain the next token is `,` — so you're injecting randomness precisely where the
model was sure. Top-p adapts.

### 7. THE PUNCHLINE — degenerate repetition

**Ask him to predict this before you tell him:** *"Greedy always picks the most likely token. That
sounds like it should give the best output. Does it?"*

Most people say yes. It's locally optimal at every step.

**It can enter a cycle it can never leave.** It picks token A, which makes B the likeliest, which makes
A the likeliest again — **forever**. Because (Concept 3) the model's output is fed back as its input, a
self-reinforcing state is a **trap**. It has no memory that it's stuck and no mechanism to escape.

**His toy shows it verbatim.** Greedy generation locks into:

```
(`tailscale (`tailscale (`tailscale (`tailscale (`tailscale...
```

...and never escapes.

- **Top-p** usually escapes — but *probabilistically*. It's a coin flip that happens to keep landing
  right. It is not a guarantee.
- A **repetition-aware sampler** (down-weight anything emitted recently) escapes **by construction**.

**Now show him the tuning subtlety, because it teaches that this is a real engineering surface and not
a cosmetic knob:**

| look-back window | result |
|---|---|
| 12 | The tight loop dies — but a **longer** cycle forms *outside the window* |
| 40 | Over-penalising builds **new** attractors. Also loops. |
| **24** | Clean. |

**Sampling policy is a correctness surface, not a style preference.** Make him say that back to you.

### 8. How a neural TTS actually works (enough of it)

```
text → [autoregressive model] → audio tokens → [vocoder/decoder] → waveform
```

An autoregressive model emits **semantic/acoustic tokens** — exactly as an LLM emits word tokens.
Then a **vocoder** turns those tokens into an actual waveform (an array of int16 samples — hello,
Module 1).

**The connection that makes his whole system make sense:**

> **A neural TTS has the same prefill/decode/sampling behaviour as an LLM — because at its core it
> *is* one.**

Everything he just learned about latency and repetition loops applies to his TTS. That is not an
analogy. It is the same machine.

Also mention **zero-shot voice cloning**: give the model a few seconds of reference audio and it
conditions on that speaker — no retraining. That's how his avatar speaks in a specific person's voice.

### 9. What vLLM is — and why it's dangerous

**vLLM** is a high-performance inference server (continuous batching, paged attention). It made his
TTS dramatically faster.

**Now flag the trap, and connect it to Module 2:** vLLM is fundamentally a **throughput** optimisation
— it exists to serve many requests efficiently. His product is a **latency** product with one user.
(Module 2: throughput and latency are different and trade off.) It still helped him, but *for reasons
he should be able to articulate.*

And the thing that actually bit him:

> **vLLM is a *different implementation* of the same model.**

Hold that sentence. It is the bug.

---

## The blanks he must fill — DO NOT WRITE THESE FOR HIM

**Blank #1 — greedy.** Return the highest-scoring next character.
*Stuck?* "You have a dict of `{char: count}`. You want the char with the biggest count. What's the
one-liner?"

**Blank #2 — top-p nucleus.** Walk the sorted candidates, accumulating `count/total`, until the running
sum reaches p. Keep those.
*Stuck?* Walk one concrete example on paper with him: probabilities `[0.5, 0.25, 0.15, 0.07, 0.03]`,
p=0.9. Which are in the nucleus? (0.5 → 0.75 → 0.90 ✓ stop. The first **three**.) Then let him code it.

**Blank #3 — sample from the nucleus, weighted by count.** (`random.choices` with `weights=`.)
*Ask:* "Why weighted, and not uniform?" (Uniform would treat a 50% token and a 15% token as equals —
you'd have thrown away the model's opinion, which is the only thing you have.)

**Blank #4 — repetition-aware.** Multiply the weight of any recently-seen char by a penalty, then take
the max.
*Ask:* "Why multiply rather than exclude outright?" (Because sometimes the repeat is *correct* —
English is full of legitimately repeated characters. You want to discourage, not forbid. Forbidding
produces gibberish.)

---

## Misconceptions — expect these exact wrong answers

| He will say | Why it's wrong | What to make him do |
|---|---|---|
| "A longer *answer* means I wait longer for the first word." | Backwards. It's the **input** length. Prefill. | The 0.016/0.109/0.813 table. Make him predict it first. |
| "Greedy is best — it always picks the most likely token." | It is the mode **most prone to degenerate looping**. Locally optimal, globally broken. | Run greedy. Watch `(`tailscale` forever. |
| "Sampling settings are a style/creativity preference." | In his system, losing one sampling rule turned a 4-second sentence into **12 seconds of dead silence**. It is a **correctness** feature. | P18. |
| "A faster implementation of the same model behaves the same." | **The big one.** See P18. | Ask what a "logits processor" is and whether a new engine would carry his over. |
| "The model knows it's repeating and will stop." | It has no such mechanism. It sees its own output as ground truth and doubles down. | Point back at the autoregressive loop. |
| "vLLM made it faster so it's strictly better." | It's a *throughput* optimisation on a *latency* product — and it silently dropped a correctness guarantee. | Module 2's throughput/latency distinction, cashed in. |

---

## Socratic question bank

- If a model makes a mistake at token 5, what does it see at token 6? Can it take it back?
- Why does a 2000-token prompt with a one-word answer feel slow?
- Why does top-p beat top-k when the model is *confident*?
- You've set temperature to 0 (i.e. greedy) "for reliability." What have you actually done to your
  risk of an infinite loop? (**Maximised it.**)
- Your TTS starts producing silence forever, mid-sentence. Where do you look first? (The sampler — not
  the audio path, not the GPU.)
- What does it mean that a text-to-speech model is "the same machine" as a language model?
- You swap in a faster inference engine and all your benchmarks improve. What could have silently
  broken?

---

## The bug this explains — "P18"

**Set it up as a mystery. Let him theorise before you reveal.**

> *The team moves the TTS's internal language model onto vLLM. Big speedup, everything benchmarks
> better. Then Chinese synthesis starts intermittently producing **12 seconds of dead silence** where
> a 4-second sentence should be. It's heard as "halting" speech. And the avatar keeps moving its mouth
> the entire time, through the silence.*
>
> *The model wasn't retrained. The text is fine. What happened?*

Likely wrong guesses (his team chased several): audio pipeline dropped buffers; the GPU stalled; the
text had bad characters; the vocoder broke.

**The answer.** vLLM's sampling pipeline **silently dropped the repetition-aware sampler** the original
implementation had. Without it, the model **looped on the silence token** — exactly the degenerate
cycle he watched happen in his toy. A 4-second sentence became ~12 seconds of dead air.

And because the mouth-generator follows the waveform it's given (Module 1!), the avatar happily
lip-synced through 12 seconds of nothing.

The fix: reimplement the repetition-aware sampler as a vLLM **logits processor**.

### The generalisation — this is the real payload

**The shape of this bug:**

> **An optimisation silently removed a correctness guarantee that nobody had written down.**

Make him feel how ordinary the mistake was. Nobody was careless. The new engine was better in every
measured way. The guarantee wasn't in a test, wasn't in a comment, wasn't in a type — it was an
*implicit property of the old implementation*, and it evaporated on contact with a new one.

**Then show him it happened AGAIN, in the same project, in a different costume.** Later they enabled
CUDA graphs on the GPU. Benchmarks got faster. **The lip-sync got worse** — because the graph capture
subtly perturbed the same sampling behaviour, degrading the audio, and the mouth follows the
waveform. The stopwatch said *better*; the human eye said *worse*.

**The eye won.** (He'll meet this again as "P33" in Module 4.)

**Ask him:** "What class of thing should you be most suspicious of when you adopt a faster
implementation?" (Answer: any behaviour that was never explicitly specified — the implicit guarantees.
And the fix isn't caution, it's *writing them down*, so the next swap has something to fail against.)

---

## Exit quiz — he must pass this before Module 4

Answers in brackets.

1. Which scales time-to-first-token: input length or output length? [**Input.** That's prefill.]
2. Why does his system split off a short opening clause? [Prefill cost scales with input length, so a
   short first piece starts speaking ~1.3 s sooner. Nothing got faster; the work was reordered.]
3. Why can greedy decoding loop forever? [Its output is fed back as input, so a self-reinforcing state
   is a trap. It has no mechanism to notice or escape.]
4. What is top-p, in one sentence? [Sample from the smallest set of tokens whose probabilities sum to
   p — a nucleus that adapts to the model's confidence.]
5. What does a repetition-aware sampler do, and why is it better than top-p at killing loops? [It
   down-weights recently-emitted tokens, so it escapes **by construction** rather than probabilistically.]
6. You swap in a faster inference engine. What class of thing might silently break? [Implicit
   correctness guarantees that were never written down — like a custom sampler.]
7. A benchmark says the change made things faster; a human says the output got worse. Who wins? [**The
   human.** The benchmark was measuring the wrong side.]

**If he misses 1, 3, or 6, do not move on.** Those three are the module.
