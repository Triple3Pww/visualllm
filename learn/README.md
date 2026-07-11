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

## Using an outside tutor

`coach/` holds one self-contained teaching brief per module — paste **one file per session** into
another AI (Gemini) to be taught that module in depth. Each repeats the rules and context, so a
fresh chat with no memory of the others still works. Start with `coach/README.md`.

**Order matters, every week:**

1. Commit that module's cold pretest guesses in `index.html` **first**.
2. Then paste `coach/0N-*.md` into the tutor and work through it.
3. Then come back here for the lab, the falsify step, and the self-check.

Guessing before you are taught is the mechanism — get taught first and you spend it for nothing.
The briefs tell the coach the same thing, and tell it **not to hand you the blanks**.

Design + evidence: `docs/superpowers/specs/2026-07-11-visualllm-study-plan-design.md`
