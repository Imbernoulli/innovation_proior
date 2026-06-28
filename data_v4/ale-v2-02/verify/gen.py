#!/usr/bin/env python3
"""Instance generator for ale-v2-02 (Rectangle Strip Packing).

Usage: python3 gen.py SEED  ->  writes one instance to stdout.

Instance format (stdin contract of the solver):
    line 1: W N
    next N lines: w_i h_i      (1 <= w_i <= W, 1 <= h_i)

A strip of fixed width W and unbounded height must hold N axis-aligned
rectangles (no rotation) packed without overlap, every rectangle fully
inside the strip horizontally (0 <= x, x + w <= W). The objective is to
minimize the used height H = max top edge over all placed rectangles.

Generation: fixed W = 1000. N is drawn from [30, 200] by seed. Each
rectangle gets a width in [1, W] and a height in [1, Hmax]; a mixture of
"thin/wide" and "tall/narrow" pieces is produced so that the packing is
non-trivial (a pure shelf packer leaves a lot of waste). All randomness is
driven by the provided integer seed so instances are reproducible.
"""
import sys
import random


def gen(seed: int):
    rng = random.Random(seed * 1000003 + 12345)
    W = 1000
    N = rng.randint(30, 200)
    Hmax = 400
    rects = []
    for _ in range(N):
        r = rng.random()
        if r < 0.34:
            # wide & short
            w = rng.randint(W // 3, W)
            h = rng.randint(1, Hmax // 4)
        elif r < 0.67:
            # tall & narrow
            w = rng.randint(1, W // 3)
            h = rng.randint(Hmax // 4, Hmax)
        else:
            # general
            w = rng.randint(1, W)
            h = rng.randint(1, Hmax)
        w = max(1, min(W, w))
        h = max(1, h)
        rects.append((w, h))
    return W, N, rects


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py SEED\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    W, N, rects = gen(seed)
    out = [f"{W} {N}"]
    for (w, h) in rects:
        out.append(f"{w} {h}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
