#!/usr/bin/env python3
"""gen.py <testId> -> ONE instance on stdout.

Power-grid substation siting (geometric-packing, weighted-radius variant).

A square service region [0,1]^2 must host N substations. Substation i has a
priority weight w_i (its served load). Each substation projects a circular
coverage disk of radius r_i; disks must lie inside the region and must not
overlap (electromagnetic-interference / clearance constraint). We maximise the
weighted coverage  sum_i w_i * r_i.

The difficulty ladder testId=1..10 grows the substation count and the weight
skew. All randomness is seeded from testId only (bit-for-bit deterministic).
Weights are drawn with a cubic transform so the distribution is heavily skewed
(a few high-load substations among many small ones): this is precisely where a
uniform grid layout is far from optimal and clever area allocation wins.
"""
import sys, random


def main():
    tid = int(sys.argv[1])
    rng = random.Random(700 + tid)

    n = 10 + tid                       # 11 .. 20 substations
    weights = []
    for _ in range(n):
        base = rng.random()
        w = 1 + int((base ** 5) * 99)  # 1 .. 100, heavily skewed toward small
        weights.append(w)

    # Guarantee at least one genuinely high-load substation for headroom.
    weights[rng.randrange(n)] = 100

    out = [str(n), " ".join(str(w) for w in weights)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
