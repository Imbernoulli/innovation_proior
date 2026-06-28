#!/usr/bin/env python3
"""Random + edge-case generator for the Longest Bitonic Subsequence problem.

Usage: gen.py <seed> [mode]
Prints a test case to stdout: first line n, second line the n values.
Keeps n small (the memoized-DFS brute is polynomial but we keep values' range tight so
collisions/equal values are common -- equal values are the tricky part, since the
subsequence must be STRICTLY increasing then STRICTLY decreasing).
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "rand"
    rng = random.Random(seed)

    if mode == "rand":
        n = rng.randint(0, 12)
        vmax = rng.choice([2, 3, 4, 5, 9, 20, 100])
        a = [rng.randint(-vmax, vmax) for _ in range(n)]
    elif mode == "tiny":
        n = rng.randint(0, 4)
        a = [rng.randint(-3, 3) for _ in range(n)]
    elif mode == "dups":
        # many equal values to stress the STRICT requirement
        n = rng.randint(1, 12)
        vmax = rng.choice([1, 2, 3])
        a = [rng.randint(0, vmax) for _ in range(n)]
    elif mode == "sorted":
        n = rng.randint(1, 12)
        a = sorted(rng.randint(-9, 9) for _ in range(n))
    elif mode == "revsorted":
        n = rng.randint(1, 12)
        a = sorted((rng.randint(-9, 9) for _ in range(n)), reverse=True)
    elif mode == "mountain":
        # a clean increase-then-decrease with some noise
        up = sorted(rng.sample(range(-20, 21), rng.randint(1, 6)))
        down = sorted(rng.sample(range(-20, 21), rng.randint(0, 6)), reverse=True)
        a = up + down
    elif mode == "plateau":
        # constant array
        n = rng.randint(1, 10)
        a = [rng.choice([-1, 0, 5])] * n
    else:
        n = rng.randint(0, 12)
        a = [rng.randint(-9, 9) for _ in range(n)]

    print(len(a))
    print(" ".join(map(str, a)))


if __name__ == "__main__":
    main()
