#!/usr/bin/env python3
"""gen.py <testId>  -- print ONE training log for the polar-base load-law problem.

Difficulty ladder (testId 1..N): larger testId => fewer training rows and higher
measurement noise, so the hidden law is harder to recover and extrapolate.

STDOUT is DATA ROWS ONLY: five whitespace-separated floats "x0 x1 x2 x3 y" per
line. The hidden law, its coefficients, the sampling seed and the held-out region
are NEVER printed here -- the ground truth lives only inside the grader.
"""
import sys, math, random


def hidden_law(x0, x1, x2, x3):
    # 4-D polynomial-exponential ground truth (kept private; the grader has its
    # own identical copy). Solvers must rediscover this FORM from the data alone.
    return 1.6 * x0 * x0 + 1.1 * x1 * x2 + 1.4 * math.exp(0.45 * x3) - 0.8 * x1 + 0.5


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        return 1
    t = int(sys.argv[1])
    if t < 1:
        t = 1

    rng = random.Random(1000 + t)
    n_rows = 80 + 20 * t            # 100 .. 280 rows
    noise = 0.06 + 0.06 * t         # nominal-regime measurement noise (grows with t)

    out = []
    for _ in range(n_rows):
        # nominal operating regime: every channel in [-1, 1]
        x0 = rng.uniform(-1.0, 1.0)
        x1 = rng.uniform(-1.0, 1.0)
        x2 = rng.uniform(-1.0, 1.0)
        x3 = rng.uniform(-1.0, 1.0)
        y = hidden_law(x0, x1, x2, x3) + rng.gauss(0.0, noise)
        out.append("%.6f %.6f %.6f %.6f %.6f" % (x0, x1, x2, x3, y))

    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
