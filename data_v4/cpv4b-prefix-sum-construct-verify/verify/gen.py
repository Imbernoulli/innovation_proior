#!/usr/bin/env python3
"""
Random small-case generator: python3 gen.py <seed>
Prints a single integer n on stdout.

Brute force enumerates n! permutations, so keep n small (<= 9 here).
We bias toward the boundary cases that matter: n = 1, small odd n
(infeasible), small even n (feasible), and powers of two (where the
tempting reverse-identity construction accidentally succeeds).
"""
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    bucket = rng.randint(0, 4)
    if bucket == 0:
        n = 1                         # trivial odd-but-feasible boundary
    elif bucket == 1:
        n = rng.choice([3, 5, 7, 9])  # odd >= 3: infeasible
    elif bucket == 2:
        n = rng.choice([2, 4, 6, 8])  # small even: feasible
    elif bucket == 3:
        n = rng.choice([2, 4, 8])     # powers of two: the trap zone
    else:
        n = rng.randint(1, 9)         # anything small

    print(n)


if __name__ == "__main__":
    main()
