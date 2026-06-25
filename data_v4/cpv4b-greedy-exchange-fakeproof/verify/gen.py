#!/usr/bin/env python3
"""Random small-case generator: python3 gen.py <seed>.

Emits a valid instance of the gondola problem:
  line 1: n C
  line 2: n weights, each 1 <= w[i] <= C  (so every climber always fits in one gondola)

Kept small so the exponential brute force stays fast, and biased toward
near-capacity weights so that pairing decisions (and the closed-form trap)
are frequently exercised.
"""
import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(0, 8)
    C = rng.randint(2, 25)

    weights = []
    for _ in range(n):
        r = rng.random()
        if r < 0.35:
            # heavy: hard to pair
            weights.append(rng.randint((C + 1) // 2 + 1, C))
        elif r < 0.7:
            # light: easy to pair
            weights.append(rng.randint(1, max(1, C // 2)))
        else:
            weights.append(rng.randint(1, C))

    print(n, C)
    if n > 0:
        print(" ".join(map(str, weights)))


if __name__ == "__main__":
    main()
