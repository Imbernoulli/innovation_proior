#!/usr/bin/env python3
"""Instance generator for "2D Rectangle Strip Packing" (ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the format:

    n W R
    w_0 h_0
    w_1 h_1
    ...
    w_{n-1} h_{n-1}

Meaning: a vertical strip of fixed integer width W and unbounded height. We are
given n axis-aligned rectangles; rectangle i has width w_i and height h_i. Each
rectangle must be placed inside the strip (0 <= x, x + width <= W, y >= 0) with
its sides parallel to the axes, and no two placed rectangles may overlap. R is a
global flag: if R == 1 a rectangle may be rotated 90 degrees (swap w and h)
before placing, if R == 0 every rectangle must keep its given orientation. The
goal is to MINIMIZE the height used, i.e. the maximum top edge over all
rectangles (see score.py / context.md for the exact rule and the feasibility
floor).

Instance regime (deterministic from the seed):
  * Strip width W in the low hundreds.
  * n rectangles in [30, 80].
  * The rectangles are produced by a recursive guillotine split of a "virtual"
    W x Htarget block, then each piece is jittered a little (and, when R == 1,
    randomly pre-rotated). So a near-perfect packing of height ~Htarget exists in
    principle, but the jitter + rotation choices make the optimum non-trivial --
    exactly the regime where placement order / rotation / best-fit-by-skyline
    matters and a plain shelf (FFDH) wastes a lot of vertical space.
"""
import sys
import random


def guillotine_pieces(rng, w, h, depth, out):
    """Recursively split a w x h block into pieces via guillotine cuts."""
    min_side = 14
    if depth <= 0 or w < 2 * min_side or h < 2 * min_side or rng.random() < 0.20:
        out.append((w, h))
        return
    if w >= h:
        cut = rng.randint(min_side, w - min_side)
        guillotine_pieces(rng, cut, h, depth - 1, out)
        guillotine_pieces(rng, w - cut, h, depth - 1, out)
    else:
        cut = rng.randint(min_side, h - min_side)
        guillotine_pieces(rng, w, cut, depth - 1, out)
        guillotine_pieces(rng, w, h - cut, depth - 1, out)


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x19A7_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    W = rng.randint(120, 220)
    # rotation allowed on roughly half the instances
    R = 1 if rng.random() < 0.5 else 0

    target_n = rng.randint(30, 80)
    # Stack a few W x band blocks so a tight packing of modest height exists.
    pool = []
    while len(pool) < target_n:
        band_h = rng.randint(int(W * 0.6), int(W * 1.4))
        guillotine_pieces(rng, W, band_h, 5, pool)
    rng.shuffle(pool)
    pool = pool[:target_n]

    rects = []
    for (w, h) in pool:
        jw = max(1, w + rng.randint(-4, 4))
        jh = max(1, h + rng.randint(-4, 4))
        if R == 1:
            # rotation is allowed: it is enough that SOME orientation fits the
            # strip. Optionally pre-rotate so the native width can exceed W.
            if min(jw, jh) > W:           # neither orientation fits -> clamp
                jw = min(jw, W)
            if rng.random() < 0.5:
                jw, jh = jh, jw
        else:
            # no rotation: the given width must fit the strip.
            jw = min(jw, W)
        rects.append((jw, jh))

    n = len(rects)
    out = [f"{n} {W} {R}"]
    out.extend(f"{w} {h}" for (w, h) in rects)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
