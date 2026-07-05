#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE training sample to stdout.

Theme: e-sports arena analytics.  A hidden "clutch-rating" law maps four
per-round player features to a scalar rating.  You are shown a TRAIN sample
drawn from the ordinary competitive range of every feature.  The hidden law,
its coefficients, and the held-out EXTRAPOLATION region live ONLY inside the
checker -- nothing about them is printed here (data rows only).

Output format (stdout):
    line 1 :  N 4            (N training rows, 4 features)
    next N :  x1 x2 x3 x4 y  (space separated floats)

Difficulty ladder (testId 1..10): larger testId  ->  fewer rows + noisier y,
so recovering a law that GENERALIZES to the extrapolation region gets harder.
"""
import sys
import random
import math


def f_clean(x1, x2, x3, x4):
    # Hidden 4-D poly-exp ground-truth law (kept identical to the checker).
    # NOTE: this function is NOT revealed through stdout -- only sampled y's are.
    return (1.5
            + 2.2 * x1 ** 2
            - 1.1 * x2
            + 1.4 * math.exp(0.6 * x3)
            + 0.9 * x1 * x4
            - 0.7 * x2 * x3
            + 0.4 * x4 ** 2)


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(2)
    t = int(sys.argv[1])
    if t < 1:
        t = 1

    n = int(200 - 15 * (t - 1))          # 200 -> 65
    if n < 40:
        n = 40
    noise = 0.15 + 0.09 * (t - 1)        # 0.15 -> 1.02
    rng = random.Random(1000 + t)        # seed derives ONLY from testId

    rows = []
    for _ in range(n):
        x = [rng.uniform(0.0, 1.0) for _ in range(4)]   # TRAIN region = [0,1]^4
        y = f_clean(*x) + rng.gauss(0.0, noise)
        rows.append((x, y))

    out = ["%d 4" % n]
    for x, y in rows:
        out.append("%.6f %.6f %.6f %.6f %.6f" % (x[0], x[1], x[2], x[3], y))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
