#!/usr/bin/env python3
"""Random test-case generator for the maximum product subarray problem.

Usage: python3 gen.py [seed]
Emits to stdout:
    n
    a[0] ... a[n-1]
with 1 <= n <= 18 and -9 <= a[i] <= 9 (matching the problem constraints).

The seed drives a mix of distributions chosen to stress the negative-sign
flip, zeros, and all-positive / all-negative corners.
"""
import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    mode = seed % 7
    if mode == 0:
        # generic small-magnitude mix (most likely to flip signs)
        n = rng.randint(1, 18)
        vals = [rng.randint(-9, 9) for _ in range(n)]
    elif mode == 1:
        # heavy on signs only, lots of negatives and zeros
        n = rng.randint(1, 18)
        vals = [rng.choice([-2, -1, 0, 1, 2]) for _ in range(n)]
    elif mode == 2:
        # all positive
        n = rng.randint(1, 18)
        vals = [rng.randint(1, 9) for _ in range(n)]
    elif mode == 3:
        # all negative
        n = rng.randint(1, 18)
        vals = [rng.randint(-9, -1) for _ in range(n)]
    elif mode == 4:
        # forced zeros sprinkled in
        n = rng.randint(1, 18)
        vals = [rng.choice([0, 0, rng.randint(-9, 9)]) for _ in range(n)]
    elif mode == 5:
        # tiny n to hit n=1,2 corners often
        n = rng.randint(1, 3)
        vals = [rng.randint(-9, 9) for _ in range(n)]
    else:
        # max length, extreme magnitudes
        n = 18
        vals = [rng.choice([-9, 9]) for _ in range(n)]

    print(n)
    print(" ".join(str(v) for v in vals))


if __name__ == "__main__":
    main()
