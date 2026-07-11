# learn/coach/ — the module briefs for an outside AI tutor

One file per module. **Paste exactly one into the coach per session.** Each is self-contained: it
repeats the learner profile, the teaching rules, and the system context, so a fresh chat with no
memory of the others still works.

| File | Module | Teaches | Ends on |
|---|---|---|---|
| `00-coach-rules.md` | — | The full method, scope, glossary. Read once yourself. | — |
| `01-sound.md` | 1 | Audio, PCM, bytes↔seconds, RMS, VAD, byte misalignment | **P40** |
| `02-streams.md` | 2 | Event loop, queues, backpressure, Little's Law, streaming | **P35** |
| `03-models.md` | 3 | Autoregression, prefill/decode, sampling, repetition loops | **P18** |
| `04-pixels.md` | 4 | GPU contention, TensorRT, CUDA graphs, the A/V sync contract | **P16 / P1** |

## The order, every week — do not shuffle this

1. **Open `learn/index.html` and commit that module's cold pretest guesses.** Two questions. Five
   minutes. You will get some wrong; that is the mechanism, not a failure.
2. **Paste that week's coach file into Gemini** and work through it. Tell it what you guessed and
   what you got wrong — that is the most useful thing it can know about you.
3. **Come back to `index.html`**: fill the toy's `TODO(you)` blanks, run it, do the falsify step,
   read the bug, answer the self-check.
4. Next week's module opens with a spaced retrieval quiz drawn from this one. Anything you missed
   comes back first.

**Why step 1 comes before step 2:** guessing before you are taught beats being taught cleanly — that
is the pretesting effect, and it holds even when the guess is wrong. Get taught first and the gated
pretest degrades into an ordinary quiz. You spend the bonus for nothing.

**Why the coach must not give you the four blanks:** filling them *is* the assessment. Each module
file says so explicitly, and tells the coach not to let you talk it into it.
