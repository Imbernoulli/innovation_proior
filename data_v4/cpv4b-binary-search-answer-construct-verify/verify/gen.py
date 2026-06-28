#!/usr/bin/env python3
"""Random small-case generator: python3 gen.py <seed>

Emits a valid instance of the aggressive-transmitters problem:
    line 1: n k
    line 2: n distinct integer candidate positions

Kept small so the C(n,k) brute force is fast, but with a spread of value ranges,
duplicates-in-spacing, and clustered configurations to exercise the off-by-one
traps in the construction.
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(2, 9)
    k = rng.randint(2, n)

    # Mix of value regimes to provoke edge behaviour. hi must be >= n-1 so the
    # universe 0..hi has at least n distinct integers to draw from.
    regime = rng.randint(0, 3)
    if regime == 0:
        hi = max(n - 1, rng.choice([5, 8, 12]))      # tight range -> many tied gaps
    elif regime == 1:
        hi = max(n - 1, rng.choice([20, 50]))        # medium
    elif regime == 2:
        hi = rng.choice([1000, 1000000])             # wide, big values (>= n-1)
    else:
        hi = max(n - 1, rng.randint(n - 1, n + 2))   # barely enough room: forces gap 1

    # Distinct positions (problem guarantees distinct candidates).
    universe = list(range(0, hi + 1))
    rng.shuffle(universe)
    pos = universe[:n]
    rng.shuffle(pos)

    print(n, k)
    print(' '.join(map(str, pos)))


if __name__ == "__main__":
    main()
