#!/usr/bin/env python3
"""Random small-case generator for fcs-gx-03. Usage: gen.py <seed>

Emits a small instance: line 1 "n k", line 2 the n positions. Kept small so
the exponential brute force stays fast. Positions can repeat (duplicate slots)
and need not be sorted. Guarantees 1 <= k <= n.
"""
import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small n so C(n,k) is tractable for the brute force.
    n = rng.randint(1, 12)
    k = rng.randint(1, n)

    # Mix of regimes: tight coordinate range (forces duplicates / zero gaps),
    # and a wider range. Occasionally make many positions identical.
    style = rng.randint(0, 3)
    if style == 0:
        hi = rng.randint(0, 5)            # tiny range -> lots of duplicates
        pos = [rng.randint(0, hi) for _ in range(n)]
    elif style == 1:
        pos = [rng.randint(0, 30) for _ in range(n)]
    elif style == 2:
        base = rng.randint(0, 50)
        pos = [base + rng.randint(0, 3) for _ in range(n)]   # clustered
    else:
        # all-equal positions stress the zero-gap path
        v = rng.randint(0, 1000000)
        pos = [v for _ in range(n)]

    rng.shuffle(pos)
    print(n, k)
    print(" ".join(map(str, pos)))


if __name__ == "__main__":
    main()
