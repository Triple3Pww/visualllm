# MODULE 2 — Streams, queues, and time

> **Paste this whole file into the coach.** It is self-contained.

---

## COACH: read this before you teach

**Who you are teaching.** Chanachon — 3rd-year ICT undergrad, on a research internship. A capable
builder: he **shipped** a real-time speech → LLM → talking-head-avatar system running locally on one
GPU. He built it with heavy AI help, and it works. But he self-assessed at **zero** in the four
foundations underneath it (audio, async, model inference, GPU). He is not a beginner programmer — he
is a builder who was never taught the layers below what he built.

He learns by watching things fail, and is impatient with theory that never cashes out.

**Three rules. Please do not break them.**

1. **He has already taken a cold pretest on this module.** Start by asking what he guessed and what he
   got wrong. Teach to those gaps first. If he hasn't done it, send him back; it's five minutes.
2. **Never write the toy's `TODO(you)` blanks for him.** Guide, ask leading questions, walk numeric
   examples — but the line comes out of *his* fingers. Filling the blanks IS the assessment.
3. **Worked examples first, not "how would you approach this?"** He is a novice here, and minimal
   guidance overloads novices (expertise-reversal effect).

**Do not teach:** threading vs multiprocessing in depth, the GIL beyond one sentence, WebRTC/ICE
internals, or asyncio's internal implementation (selectors, futures). He needs the *model*, not the
machinery.

**Time: 2–3 hours.**

**He has done Module 1** (audio is an array of int16; bytes ↔ seconds; one dropped byte = loud noise;
a test whose reference shares the suspect input cannot fail). You may build on that.

---

## The system he built (context for you)

Speech → voice-activity detection → speech-to-text → LLM → text-to-speech → talking-head avatar →
back to the browser. Every stage is a **coroutine**, and between them sit **queues**. The whole thing
**streams**: the LLM's first sentence reaches the TTS before the full answer exists, and the TTS's
first audio chunk reaches the avatar immediately. Target: first sound in under 3 seconds.

**This module is the second hop: the words becoming a stream.** It also teaches the shape of the
system's *worst measurement mistake* — which is a different and subtler class of bug than Module 1's.

---

## Session plan

| Time | What |
|---|---|
| 10 min | His pretest guesses. What did he get wrong? |
| 30 min | Concepts 1–4: concurrency, blocking, the event loop, queues |
| 40 min | Concept 5: **bounded vs unbounded, and backpressure.** The core. Make him predict first. |
| 20 min | Concept 6: Little's Law — the identity underneath it |
| 25 min | Concepts 7–9: throughput vs latency, streaming, paced release |
| 20 min | The bug (P35) — where do you stop the stopwatch? |
| 15 min | Exit quiz |

---

## The concepts, in teaching order

### 1. Concurrency is not parallelism

**Parallelism**: doing two things at literally the same instant. Requires two cores.
**Concurrency**: making progress on many things by interleaving them on one core.

His pipeline is **concurrent, mostly single-threaded** — and that is fine, because its stages spend
almost all their time **waiting** (for the network, for the GPU, for the model) rather than computing.
Waiting is not work. One thread can wait on a hundred things.

> **Analogy that works:** a chef with one pan is *parallel*-limited. A chef with one pan but four pots
> simmering is *concurrent* — he stirs whichever needs stirring. He isn't doing four things at once;
> he's never idle.

### 2. Blocking vs non-blocking

A **blocking** call (`time.sleep`, a plain socket read, a synchronous file read) stops the entire
thread. Nothing else progresses.

In a real-time audio pipeline this is fatal: if one stage blocks, every stage downstream **starves** —
and audio that arrives late is worthless, because the moment it was needed has passed.

**Make the point hard:** in a batch system, a slow call is a slow call. In a real-time system, a
blocking call is a *correctness* bug.

### 3. The event loop, coroutines, `await`

Teach it as a **to-do list**, not as machinery.

The loop holds a list of tasks. It runs one. That task runs until it hits an `await` on something not
ready yet — a network reply, a timer, a queue item — at which point it **voluntarily hands control
back**. The loop then runs someone else. When the awaited thing becomes ready, the task is resumed
where it left off.

That is 90% of `asyncio`.

**Emphasise the word *voluntarily*.** Cooperative multitasking means a task that never awaits — a
tight CPU loop, a blocking call — **starves everything else**. Nothing preempts it. This is the single
most common way people destroy an async system, and it's why "just add async" doesn't fix a CPU-bound
stage.

**Drill:** "I have an async pipeline and I call a synchronous 200 ms image resize inside one stage.
What happens to the other stages?" (They all stop for 200 ms. Every one.)

### 4. Producer / consumer queues

Stages connect through queues. Stage A **puts**, stage B **gets**. The queue **decouples their
speeds**.

That decoupling is the entire point of a queue — and, as he's about to discover, its entire danger.

### 5. THE CORE LESSON — bounded vs unbounded, and backpressure

**Set the scenario up and make him predict before you reveal anything.**

> A producer makes an item every **50 ms**. A consumer takes **120 ms** per item. The consumer is
> 2.4× too slow. The queue between them has **no size limit**.
>
> Nothing crashes. Nothing is dropped. Every item eventually comes out, in order, correct.
>
> **So what goes wrong?**

Let him sit with it. He will probably say "memory" — that's true but it's not the interesting answer,
and in a short conversation it never actually blows up.

**The answer he must reach: LATENCY.**

The queue absorbs the speed mismatch by **getting longer**. So by the end of the turn you are
delivering something that was made *seconds ago*. The system looks healthy — no errors, no drops — and
is producing stale output.

> **An unbounded queue does not fix a slow consumer. It HIDES it, and bills you in staleness.**

Now bound the queue to 2 items. Now `await queue.put()` **cannot complete** when the queue is full —
so it **blocks the producer**, forcing the whole pipeline down to the speed of its slowest stage.

That blocking is called **backpressure**, and it is a **feature**.

**The measured result from his toy** (make him predict each number first):

| | worst item age | total wall time |
|---|---|---|
| unbounded | **1.30 s** | 2.52 s |
| bounded (2) | **0.45 s** | 2.56 s |

**Nearly 3× fresher. Identical total time.**

Sit on that. Almost everyone predicts this backwards — they expect that deliberately throttling the
producer must make the system *slower*. It didn't cost a thing. **Throttling the producer cost nothing
in throughput and bought everything in latency.**

Ask him why the total time is unchanged. (Because the *consumer* was always the bottleneck. The
producer running ahead never made the consumer any faster — it just built a backlog. The pipeline's
throughput was always 1/120ms, no matter what the producer did.)

### 6. Little's Law — the identity underneath all of it

Give him this. It converts a rule of thumb into mathematics.

```
L = λ × W

L = items in the system    λ = arrival rate    W = time in system
```

Rearranged: **`W = L / λ`**

**Time-in-system is proportional to queue length.** At a fixed throughput, a longer queue *is* more
latency. Not "tends to be" — *is*. It's an identity, not a heuristic.

This is why **"just make the buffer bigger"** is so often exactly the wrong instinct. A bigger buffer
does not make anything faster; it makes the backlog *legal*.

**Drill:** "Consumer handles 10 items/sec. There are 30 items in the queue. How long until a new
arrival is served?" → `W = 30/10` = **3 seconds**. He should be able to do this instantly.

### 7. Throughput vs latency

They are **different**, they **trade off**, and optimising one can wreck the other.

- **Throughput**: items per second. A highway's lanes.
- **Latency**: time for *one* item. How long *your* trip takes.

**Batching raises throughput and raises latency.** A bus carries more people per hour than a taxi, and
every passenger waits longer.

Flag it: **this distinction returns hard in Module 3**, where the fast inference server he uses (vLLM)
is fundamentally a throughput optimisation, and his product is a latency product.

### 8. Streaming

The LLM does not produce a complete answer and hand it over. It emits **one token at a time**.

So: the first **sentence** can be sent to the TTS before the rest of the answer exists. And the first
**audio chunk** can be sent to the avatar before the sentence has finished synthesising.

**The consequence he must own:**

> **Time-to-first-output depends on the FIRST PIECE, not on the total length.**

His system exploits this deliberately. It splits off a **short opening clause** and synthesises that
first — which cut time-to-first-sound from **~4.6 s to ~3.2 s**. The full answer takes exactly as long
as it always did. The *user's wait* collapsed.

Ask him: "why does that work? Nothing got faster." (Nothing did. The work was **reordered** so the
part the user is waiting on happens first. That's a latency win with zero throughput win — and it's
free.)

*Module 3 explains **why** a short first clause is so much cheaper. Plant the question; don't answer
it.*

### 9. Paced release vs free-run

Two ways to emit something:

- **Free-run:** ship it the instant it's ready.
- **Paced:** item N leaves at `t0 + N/fps`, on a **fixed clock**, regardless of when it arrived.

He builds the paced release clock in this module's toy. **Flag it hard: this is the central design
decision of Module 4.** Right now it's just a clock. In four weeks it'll be the reason his avatar's
lips match the words.

### 10. Where do you stop the stopwatch?

Set up the bug below with a question, before revealing anything:

> "You want to measure how long the user waits to hear a reply. Where, exactly, does the timer stop?"

Let him answer. Then push: "Are you *sure* that's where the user is?"

---

## The blanks he must fill — DO NOT WRITE THESE FOR HIM

**Blank #1 — put the item on the queue.**
He must use the **async** put (`await out.put(...)`), not `put_nowait`.
*This matters enormously:* `put_nowait` never blocks, so the bounded case would silently behave like
the unbounded one, and the entire lesson evaporates. **If he uses `put_nowait`, don't just correct
him — ask him to predict what the bounded run will now show, then let him run it and find out.** That
mistake is worth more than the correct answer.

**Blank #2 — compute the release deadline** for item n: `start + n / FPS`.

**Blank #3 — sleep only if early.** Never sleep a negative duration.
*Ask:* "What happens if you're late and you sleep the negative difference?" (Either it throws, or it
returns instantly — but the real point is that a late item must go **immediately**, not be delayed
further. You can't un-fall-behind by waiting.)

---

## Misconceptions — expect these exact wrong answers

| He will say | Why it's wrong | What to make him do |
|---|---|---|
| "A bigger buffer is safer." | Little's Law: a bigger buffer **is** more latency at fixed throughput. | `W = L/λ`. Make him compute it. |
| "Throttling the producer will slow the system down." | Measured: **identical** wall time, 3× fresher. The consumer was always the bottleneck. | Have him predict both numbers, then run the toy. |
| "Latency is just 1/throughput." | No. A system can have enormous throughput and dreadful latency. That's what batching *is*. | The bus vs taxi. |
| "It didn't crash and nothing was dropped, so it's fine." | The unbounded queue never crashed. It was delivering seconds-stale output. | "Healthy" and "correct" are not the same. |
| "The server logged that it started speaking, so the user heard it." | Off by **1.26 seconds**. See below. | P35. |
| "async makes things faster." | async makes *waiting* cheap. It does nothing for CPU-bound work — and a blocking call in an async loop starves everything. | The 200 ms image-resize drill. |

---

## Socratic question bank

- Why can one thread handle a hundred network connections, but not a hundred matrix multiplications?
- Your pipeline has 5 stages. Stage 3 takes 500 ms; the rest take 10 ms. What is the pipeline's
  throughput? (~2/sec — the slowest stage sets it. Everything else is idle.)
- If you double the size of every queue in a healthy system, what happens? (Throughput: nothing.
  Latency: worse, under load. You bought yourself the *right* to fall further behind.)
- The queue is always empty. Is that good or bad? (Good — it means the consumer keeps up. An always-
  empty queue is a *healthy* queue. An always-full one is a bottleneck with a waiting room.)
- Why does splitting off a short first clause improve the user's experience when nothing got faster?
- Name a metric you've seen in your own project that stops before the user does.

---

## The bug this explains — "P35"

**Set it up as a mystery.**

> *The team has a metric: "time to first output." They've been optimising against it for weeks, and
> it says 2.9 seconds. Under their 3-second goal. But when a human sits down and uses the thing, it
> feels slower than the number says. The number isn't lying — every measurement is correct. What's
> going on?*

**The answer.** The metric measured the gap between "user stopped speaking" and a `BotStartedSpeaking`
event — an event emitted **inside the server**.

But the user's ear is on the far side of:

- the transport layer,
- a video/audio **encode**,
- a **network** hop,
- and the browser's **jitter buffer**.

All of which happens **after the stopwatch stopped**. It was worth **1.26 seconds**.

So they spent weeks tuning hard against a number that did not describe the experience they were
shipping. **Every value it reported was true.** The thing it measured simply wasn't the thing that
mattered.

The fix was to stitch the *client's* arrival clock onto the server's start time and report a full
waterfall **to the ear**.

**Make him state the lesson.** What you're fishing for:

> *A metric that stops before the user does is not a metric — it is a comfort.*

Then the follow-up that generalises it: **"Where else in your system does a measurement stop short of
the user?"** (Good answers: server-side render time that ignores transport; "model latency" that
ignores tokenisation; any benchmark run without the GPU contention that exists in production.)

Note the family resemblance to Module 1's P40: **both bugs are failures of *metrology*, not of code.**
In P40 the test couldn't fail. In P35 the metric measured the wrong thing. The code was fine both
times. Ask him what that suggests about where to spend his suspicion.

---

## Exit quiz — he must pass this before Module 3

Answers in brackets.

1. One word: what does a bounded queue do to a producer that outruns its consumer? [Blocks it]
2. What is that called, and is it a bug or a feature? [Backpressure. A feature.]
3. State Little's Law and use it to argue against making a buffer bigger.
   [`L = λW`, so `W = L/λ`. At fixed throughput, more items queued = proportionally more wait. A
   bigger buffer buys the *right* to fall further behind.]
4. True or false: time-to-first-output depends on the length of the whole reply. [**False** — it
   depends on the first piece. That's what streaming buys you.]
5. Your consumer handles 10 items/sec and 30 are queued. How long until a new arrival is served? [3 s]
6. Why can a metric be perfectly honest and still be wrong? [Because it can measure the wrong thing.
   Every value true, every value irrelevant. It stopped before the user did.]
7. A blocking call inside one async stage — what happens to the other stages? [They all stop. Nothing
   preempts a task that doesn't await.]

**If he misses 1, 3, or 6, do not move on.** Those three are the module.
