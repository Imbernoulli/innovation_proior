#!/usr/bin/env python3
"""gen.py <testId> -> one instance on stdout.

Difficulty ladder: testId 1..10 maps to odd N = 7 + 2*testId (N = 9,11,...,27).
The lead-drone beacon pattern r is a deterministic +/-1 vector seeded by testId only.
All randomness is seeded; no wall-time / no external entropy.
"""
import sys
import random


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    if tid < 1:
        tid = 1
    if tid > 10:
        tid = 10
    N = 7 + 2 * tid  # odd: 9..27

    rnd = random.Random(90000 + 7919 * tid)  # seed depends on testId ONLY
    r = [rnd.choice([-1, 1]) for _ in range(N)]

    out = [str(N), " ".join(str(x) for x in r)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
