#!/usr/bin/env python3
"""Instance generator for the museum-gallery-tour uniform-spread problem.

    python3 gen.py <testId>

prints ONE instance to stdout. testId = 1..N is a difficulty ladder: the number
of free viewing stations grows with testId. All randomness is seeded ONLY by the
testId, so instances are bit-for-bit reproducible.

Instance format (stdout):
    line 1 : "n k"         n = free stations to place, k = fixed landmarks
    next k : "x y"         landmark coordinates in [0,1]^2
"""
import sys


def landmarks(t, k):
    import random
    rng = random.Random(1000 + t)
    pts = []
    tries = 0
    while len(pts) < k and tries < 200000:
        tries += 1
        x = rng.uniform(0.12, 0.88)
        y = rng.uniform(0.12, 0.88)
        # keep landmarks off the main diagonal (so the checker's diagonal
        # baseline construction has a well-defined, non-degenerate d_min)
        if abs(x - y) < 0.12:
            continue
        # keep landmarks mutually well separated so a fixed landmark pair is
        # never the binding closest pair for a good layout
        if all((x - a) ** 2 + (y - b) ** 2 >= 0.25 ** 2 for a, b in pts):
            pts.append((x, y))
    # fallback (should not trigger for the sizes used): deterministic spread
    while len(pts) < k:
        i = len(pts)
        pts.append((0.20 + 0.30 * i, 0.85 - 0.30 * i))
    return pts


def main():
    t = int(sys.argv[1])
    n = 6 + t          # free stations: 7 .. 16
    k = 3              # fixed landmarks (entrance, exit, existing sculpture)
    pts = landmarks(t, k)
    out = ["%d %d" % (n, k)]
    for x, y in pts:
        out.append("%.6f %.6f" % (x, y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
