#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE tide-pool instance to stdout.

testId 1..10 is a difficulty ladder in the number of pools n (small -> large).
Everything is seeded by testId only, so generation is deterministic.

Instance format (stdout):
  line 1: n S          # number of pools, and the exact total population to place
  line 2: cap_1 ... cap_n   # integer carrying-capacity of each pool
"""
import sys, random


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rng = random.Random(1000 + t)

    # difficulty ladder: number of tide pools grows with testId (large-scale family)
    n = 30 + (t - 1) * 16          # t=1 -> 30 ,  t=10 -> 174

    # integer carrying capacity of each pool
    cap = [rng.randint(4, 10) for _ in range(n)]

    # total population to distribute; kept well below sum(cap) so a flat spread
    # never saturates any single pool (level ~ 0.40*mean(cap) < min cap = 4).
    S = int(round(0.40 * sum(cap)))
    if S < 1:
        S = 1

    out = [f"{n} {S}", " ".join(str(c) for c in cap)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
