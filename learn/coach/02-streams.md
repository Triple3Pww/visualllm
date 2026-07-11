# Module 2 — Streams, queues, and time

## Who I am and what I'm doing

I'm Chanachon, a 3rd-year ICT student at Mahidol, on a research internship. I built a real-time system
where you speak into a browser and a photoreal talking-head avatar answers you out loud: microphone →
detect speech → speech-to-text → LLM → text-to-speech → face-animation model → back to the browser. It
runs on one GPU. I built it with heavy AI help and I'm now learning the fundamentals underneath, one
layer per week. (Last week: audio is an array of int16 numbers, 2 bytes each, and one dropped byte turns
speech into loud noise.)

**This module is about how the stages connect and stream, and why timing is everything.** Teach me this
content, then quiz me at the bottom. I have a small Python exercise after, so explain ideas, don't write
code for me.

---

## What I need to learn

### Concurrency is not parallelism
**Parallelism** = two things at literally the same instant (needs two cores). **Concurrency** = making
progress on many things by interleaving them on one core. My pipeline is concurrent and mostly
single-threaded, and that's fine, because the stages spend almost all their time *waiting* (for the
network, the GPU, the model), not computing. One thread can wait on a hundred things. (A chef with one
pan but four simmering pots: he's never idle, but he's not cooking four things at once.)

### Blocking vs non-blocking
A **blocking** call (like `time.sleep`, or a plain synchronous read) stops the whole thread. In a
real-time audio pipeline that's fatal: if one stage blocks, everything downstream starves, and audio
that arrives late is worthless. In a batch program a slow call is just slow; in a real-time system a
blocking call is a *correctness* bug.

### The event loop
Think of it as a to-do list. The loop runs one task until it hits an `await` on something not ready yet
(a network reply, a timer, a queue item) — at which point the task **voluntarily hands control back**,
and the loop runs someone else. When the awaited thing is ready, the task resumes. That's 90% of
`asyncio`. The catch: it's *cooperative* — a task that never awaits (a tight CPU loop, a blocking call)
starves everything, because nothing preempts it.

### Producer/consumer queues
Stages connect through queues: stage A puts, stage B gets. The queue **decouples their speeds**. That's
its whole purpose — and its whole danger.

### The core idea — bounded queues and backpressure
Scenario: a producer makes an item every 50 ms; a consumer takes 120 ms per item (2.4× too slow); the
queue has no size limit. Nothing crashes, nothing is dropped. **What goes wrong?**

**Latency.** The queue absorbs the mismatch by growing, so by the end you're delivering something made
seconds ago. An unbounded queue doesn't fix a slow consumer — it *hides* it and bills you in staleness.

Now bound the queue to 2 items. `await queue.put()` can't finish when the queue is full, so it **blocks
the producer**, forcing the whole pipeline down to the speed of its slowest stage. That's
**backpressure**, and it's a feature. In my exercise, bounding the queue drops the worst item's staleness
from **1.30 s to 0.45 s** (~3× fresher) while total time is **identical** (2.52 s vs 2.56 s). Throttling
the producer cost nothing and bought everything — because the consumer was always the bottleneck.

### Little's Law
```
L = λ × W     (items in system = arrival rate × time in system)
→ W = L / λ
```
Time-in-system is proportional to queue length. At fixed throughput, a longer queue *is* more latency —
it's an identity, not a guess. This is why "just make the buffer bigger" is usually wrong: a bigger
buffer doesn't make anything faster, it just makes the backlog legal.

### Throughput vs latency
Different things that trade off. **Throughput** = items/sec (a highway's lanes). **Latency** = time for
one item (how long your trip takes). Batching raises throughput *and* latency — a bus moves more people
per hour than a taxi, and every passenger waits longer. (This comes back hard in Module 3.)

### Streaming
The LLM emits one token at a time, so the first *sentence* reaches the TTS before the full answer exists,
and the first *audio chunk* reaches the avatar before the sentence finishes. Consequence:
**time-to-first-output depends on the first piece, not the whole reply.** My system deliberately splits
off a short opening clause and speaks it early — that cut time-to-first-sound from ~4.6 s to ~3.2 s. The
full answer takes exactly as long as before; the *wait* collapsed, because the work was reordered so the
part I'm waiting on happens first. (Module 3 explains *why* a short first clause is so much cheaper.)

### Paced release vs free-run
Two ways to emit something: **free-run** = ship it the instant it's ready; **paced** = item N leaves at
`t0 + N/fps`, on a fixed clock, no matter when it arrived. Remember paced release — it turns out to be the
central design decision of Module 4.

---

## The bug this explains ("P35" in my project)

I had a metric — "time to first output" — and it read 2.9 s, under my 3-second goal. But when a human
actually used the system, it felt slower than the number said, and the number wasn't lying.

The metric measured up to a "bot started speaking" event emitted **inside the server**. But my ear is on
the far side of a transport layer, a video/audio encode, a network hop, and the browser's jitter buffer —
all of which happen *after* the stopwatch stopped. That was worth **1.26 seconds**. I'd been tuning hard
against a number that didn't describe the experience I was shipping. Every value it reported was true; it
just measured the wrong thing. Lesson: **a metric that stops before the user does is not a metric, it's a
comfort.** (Note the family resemblance to last week's bug — both were failures of *measurement*, not code.)

---

## Questions for Gemini to ask me

1. What's the difference between concurrency and parallelism, and which is my pipeline?
2. Why is a blocking call inside one async stage a problem for *all* the stages?
3. A fast producer feeds a slow consumer through an unbounded queue. Nothing crashes. What goes wrong?
4. What is backpressure, and is it a bug or a feature?
5. State Little's Law and use it to argue against making a buffer bigger.
6. My consumer handles 10 items/sec and 30 are queued — how long until a new arrival is served?
7. True or false: time-to-first-output depends on the length of the whole reply. Why?
8. Why does splitting off a short first clause help, when nothing actually got faster?
9. How can a metric be perfectly honest and still be wrong?
10. Where else in a system might a measurement stop short of the user?

If I miss 4, 5, or 9, make me work through it again — those are the heart of this module.
