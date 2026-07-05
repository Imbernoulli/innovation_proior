#!/usr/bin/env python3
"""gen.py <testId>  -- print ONE noisy strong-lensing profile for the hidden
radial force-law recovery problem (format E).

Astrophysics / gravitational-lensing setting: a cluster-scale lens produces a
radial deflection (force) profile F(r).  The instrument samples F only in the
INNER strong-lensing band (LOW r), where a short-range structural correction is
non-negligible.  The solver must rediscover the closed-form law F(r) so that it
EXTRAPOLATES to the OUTER weak-lensing band (HIGH r), which the grader holds out.

Difficulty ladder (testId 1..N): larger testId => fewer measured radii and larger
relative measurement noise, so the hidden law is harder to pin down and extrapolate.

STDOUT is DATA ROWS ONLY: two whitespace-separated floats "r F" per line.  The
hidden law, its coefficients, the sampling seed and the held-out (outer) band are
NEVER printed here -- the ground truth lives only inside the grader.
"""
import sys, random


def hidden_law(r):
    # Radial force / deflection law (kept private; the grader keeps an identical
    # copy).  Leading inverse-square term A/r^2 plus a STEEP short-range structural
    # correction C/r^4 that only matters at small r.  Solvers must rediscover this
    # FORM from the inner-band data alone.
    A = 2.0
    C = 0.06
    return A / (r * r) + C / (r * r * r * r)


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        return 1
    t = int(sys.argv[1])
    if t < 1:
        t = 1

    rng = random.Random(4200 + t)
    n_rows = 160 - 12 * t             # 148 .. 40 rows
    noise = 0.03 + 0.012 * t          # relative measurement noise (grows with t)

    out = []
    for _ in range(n_rows):
        # INNER strong-lensing band only: r in [0.15, 0.8]
        r = rng.uniform(0.15, 0.8)
        f = hidden_law(r) * (1.0 + rng.gauss(0.0, noise))
        out.append("%.6f %.6f" % (r, f))

    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
