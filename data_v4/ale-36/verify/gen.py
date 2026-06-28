#!/usr/bin/env python3
"""Instance generator for single-machine total weighted tardiness (1 || sum w_j T_j).

Usage: python3 gen.py SEED  ->  writes an instance to stdout.

We follow the standard OR-Library / Crauwels-Potts-Van Wassenhove style generation
for weighted-tardiness benchmarks:
  - n jobs.
  - processing time p_j ~ U[1, 100].
  - weight        w_j ~ U[1, 10].
  - due dates depend on the total processing P = sum p_j via two control knobs:
      RDD (relative range of due dates) and TF (average tardiness factor),
      drawn per instance from a discrete grid so different seeds give different
      "hardness" regimes:
        d_j ~ U[ P*(1 - TF - RDD/2), P*(1 - TF + RDD/2) ], clamped to >= 1.
    Small RDD / large TF => tight, bunched due dates (hard, lots of tardiness).

Everything is integer. The instance is fully determined by SEED.
"""
import sys, random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rng = random.Random(seed * 1000003 + 17)

    # problem size: vary in a band so the seed set spans small..moderate n.
    n = rng.choice([40, 50, 60, 75, 90, 100])

    p = [rng.randint(1, 100) for _ in range(n)]
    w = [rng.randint(1, 10) for _ in range(n)]
    P = sum(p)

    RDD = rng.choice([0.2, 0.4, 0.6, 0.8, 1.0])
    TF = rng.choice([0.2, 0.4, 0.6, 0.8])

    lo = P * (1.0 - TF - RDD / 2.0)
    hi = P * (1.0 - TF + RDD / 2.0)
    if hi < lo:
        lo, hi = hi, lo
    d = []
    for _ in range(n):
        val = rng.uniform(lo, hi)
        dj = int(round(val))
        if dj < 1:
            dj = 1
        d.append(dj)

    out = [str(n)]
    for j in range(n):
        out.append(f"{p[j]} {w[j]} {d[j]}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
