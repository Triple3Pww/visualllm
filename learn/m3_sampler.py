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
RAS_WINDOW = 24    # look back this far when penalising repeats. tuned: 12 is too
                   # short (a LONGER cycle just forms outside the window) and 40 is
                   # too aggressive (over-penalising builds new attractors). try it.
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
    # The Windows console is cp1252 and the corpus has Unicode in it, so a generated
    # arrow or em-dash would crash the print. pipeline/main.py does exactly this too.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
