#!/usr/bin/env python3
"""Random + edge-case generator for the OBST problem.

Usage: gen.py <seed> [mode]
  mode in {tiny, small, mid, edge, big}; default chosen from seed.
Prints a valid stdin instance: n then n frequencies.
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else None
    rng = random.Random(seed)

    if mode is None:
        mode = rng.choice(["tiny", "tiny", "small", "small", "mid", "edge"])

    if mode == "edge":
        # deterministic-ish edge instances keyed by seed
        choice = seed % 8
        if choice == 0:
            print(0)                       # n = 0
            return
        elif choice == 1:
            print(1)                       # single key
            print(rng.randint(0, 1000))
            return
        elif choice == 2:
            n = rng.randint(2, 8)
            print(n)
            print(" ".join("0" for _ in range(n)))   # all-zero frequencies
            return
        elif choice == 3:
            n = rng.randint(2, 8)
            print(n)
            v = rng.randint(1, 1000)
            print(" ".join(str(v) for _ in range(n)))  # all equal
            return
        elif choice == 4:
            n = rng.randint(2, 8)
            print(n)
            # strictly increasing
            print(" ".join(str(k * 7 + 1) for k in range(n)))
            return
        elif choice == 5:
            n = rng.randint(2, 8)
            print(n)
            # strictly decreasing
            print(" ".join(str((n - k) * 7 + 1) for k in range(n)))
            return
        elif choice == 6:
            n = rng.randint(2, 8)
            print(n)
            # one huge frequency in the middle (tempts 'most frequent at root')
            f = [rng.randint(1, 5) for _ in range(n)]
            f[n // 2] = 10**9
            print(" ".join(str(x) for x in f))
            return
        else:
            n = rng.randint(2, 8)
            print(n)
            # spike at an end
            f = [rng.randint(1, 5) for _ in range(n)]
            f[0] = 10**9
            print(" ".join(str(x) for x in f))
            return

    if mode == "tiny":
        n = rng.randint(0, 6)
        hi = rng.choice([3, 5, 10, 50])
    elif mode == "small":
        n = rng.randint(1, 8)
        hi = rng.choice([10, 100, 1000])
    elif mode == "mid":
        n = rng.randint(1, 11)
        hi = rng.choice([100, 1000, 10**6])
    else:  # big — for sol-only timing, brute will be too slow; not used in diff test
        n = rng.randint(400, 500)
        hi = 10**9

    print(n)
    print(" ".join(str(rng.randint(0, hi)) for _ in range(n)))


if __name__ == "__main__":
    main()
