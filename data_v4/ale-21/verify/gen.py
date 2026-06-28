#!/usr/bin/env python3
"""Instance generator for "Guillotine Cutting Stock" (ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the format:

    n W H
    w_0 h_0
    w_1 h_1
    ...
    w_{n-1} h_{n-1}

Meaning: there is an unlimited supply of identical rectangular sheets of size
W x H. We are given n requested rectangles; rectangle i has width w_i and height
h_i. Each requested rectangle may be placed (axis-aligned, optionally rotated 90
degrees) onto some sheet, and the placement on every sheet must be realizable by
guillotine (edge-to-edge) cuts. The goal is to use few sheets and waste little
area; unplaced rectangles are allowed but penalized (see score.py / context.md).

Instance regime (deterministic from the seed):
  * Sheet size W, H in a few hundred range.
  * n requested rectangles in [40, 90].
  * Each rectangle is produced by a recursive guillotine split of a "virtual"
    template sheet, then jittered, so the instance is dense (a near-perfect
    packing exists in principle) but the jitter + rotation choices make the
    optimum non-trivial -- exactly the regime where ordering/best-fit heuristics
    matter and a trivial next-fit shelf wastes a lot.
"""
import sys
import random


def guillotine_pieces(rng, w, h, depth, out):
    """Recursively split a w x h area into pieces via guillotine cuts."""
    # Stop splitting with some probability or when too small.
    min_side = 18
    if depth <= 0 or w < 2 * min_side or h < 2 * min_side or rng.random() < 0.22:
        out.append((w, h))
        return
    if w >= h:
        # vertical cut
        lo = min_side
        hi = w - min_side
        cut = rng.randint(lo, hi)
        guillotine_pieces(rng, cut, h, depth - 1, out)
        guillotine_pieces(rng, w - cut, h, depth - 1, out)
    else:
        # horizontal cut
        lo = min_side
        hi = h - min_side
        cut = rng.randint(lo, hi)
        guillotine_pieces(rng, w, cut, depth - 1, out)
        guillotine_pieces(rng, w, h - cut, depth - 1, out)


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x6311_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    # Sheet size.
    W = rng.randint(180, 300)
    H = rng.randint(180, 300)

    # Build a pool of pieces by guillotine-splitting a few template sheets, so a
    # good packing genuinely exists; then keep n of them.
    target_n = rng.randint(40, 90)
    pool = []
    while len(pool) < target_n + 20:
        guillotine_pieces(rng, W, H, 5, pool)
    rng.shuffle(pool)
    pool = pool[:target_n]

    # Jitter each piece a little and clamp to the sheet, and randomly swap w/h.
    rects = []
    for (w, h) in pool:
        jw = max(1, w + rng.randint(-6, 6))
        jh = max(1, h + rng.randint(-6, 6))
        if jw > W:
            jw = W
        if jh > H:
            jh = H
        if rng.random() < 0.5:
            jw, jh = jh, jw
            if jw > W:
                jw = W
            if jh > H:
                jh = H
        rects.append((jw, jh))

    n = len(rects)
    out = [f"{n} {W} {H}"]
    out.extend(f"{w} {h}" for (w, h) in rects)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
