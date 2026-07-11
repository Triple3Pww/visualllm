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
