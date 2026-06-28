#!/usr/bin/env python3
"""Random + edge-case generator for the subset-sum-count problem.

Usage: gen.py SEED [MODE]

Prints a test case in the stdin format:
    n T
    a[0] ... a[n-1]

Keeps n small (<= ~18) so the 2^n brute oracle stays feasible, while still
exercising zeros, duplicates, T just past the max sum, T = 0, etc.
"""
import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "rand"
    rng = random.Random(seed)

    if mode == "rand":
        n = rng.randint(0, 16)
        vmax = rng.choice([0, 1, 2, 3, 5, 10, 30])
        a = [rng.randint(0, vmax) for _ in range(n)]
        total = sum(a)
        # Bias T to be reachable a good fraction of the time.
        if rng.random() < 0.7 and total > 0:
            T = rng.randint(0, total)
        else:
            T = rng.randint(0, total + 5)
    elif mode == "zeros":
        # Lots of zeros: every zero doubles the count for a fixed sum.
        n = rng.randint(0, 16)
        a = [0] * n
        k = rng.randint(0, n)
        for i in range(k):
            a[i] = rng.randint(1, 4)
        rng.shuffle(a)
        T = rng.randint(0, sum(a) + 2)
    elif mode == "dups":
        # Many equal positive values: distinct positions => distinct subsets.
        n = rng.randint(0, 16)
        val = rng.randint(1, 3)
        a = [val] * n
        T = rng.randint(0, val * n + 2)
    elif mode == "tzero":
        # Force T = 0: answer is 2^(number of zeros).
        n = rng.randint(0, 16)
        a = [rng.randint(0, 3) for _ in range(n)]
        T = 0
    elif mode == "edge_empty":
        print("0 0")
        return
    elif mode == "edge_tbig":
        n = rng.randint(1, 16)
        a = [rng.randint(0, 4) for _ in range(n)]
        T = sum(a) + rng.randint(1, 4)  # unreachable
        print(f"{n} {T}")
        print(" ".join(map(str, a)))
        return
    else:
        n = 0
        a = []
        T = 0

    print(f"{n} {T}")
    print(" ".join(map(str, a)))


if __name__ == "__main__":
    main()
