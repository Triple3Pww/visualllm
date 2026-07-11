# learn/coach/ — study material to hand to Gemini

One file per module. **Paste one file per session into Gemini.** Each is self-contained: it starts with
who I am and what I built, then the plain content to learn, then the real bug it explains, then the
questions I want Gemini to quiz me with.

| File | Module | Content | Bug it explains |
|---|---|---|---|
| `01-sound.md` | 1 | Audio, PCM, bytes↔seconds, RMS, VAD, byte misalignment | P40 |
| `02-streams.md` | 2 | Event loop, queues, backpressure, Little's Law, streaming | P35 |
| `03-models.md` | 3 | Autoregression, prefill/decode, sampling, repetition loops | P18 |
| `04-pixels.md` | 4 | GPU contention, TensorRT, CUDA graphs, the A/V sync contract | P16 / P1 |

After a module, come back to `learn/index.html` to do that module's small Python exercise and answer its
self-check. One module a week; Module 3 is the hard one — give it two sessions if you need to.
