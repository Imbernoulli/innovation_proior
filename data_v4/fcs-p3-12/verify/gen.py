#!/usr/bin/env python3
"""
Random + edge test generator for the no-k-consecutive-ones counting problem.

Usage: python3 gen.py <seed>

Emits stdin in the judged format:
    T
    N k p   (T lines)

The brute oracle is O(N*k), so this generator keeps N modest (<= ~2000) while
still exercising many (N, k, p) combinations, plus deliberate edge cases:
N=0, N<k (the "tidy 2^N" regime that tempts hardcoding), k=1, k=2, small primes
and composite moduli, and N exactly at the recurrence boundary.
"""
import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    cases = []

    # A pile of random small/medium cases.
    for _ in range(40):
        k = rng.randint(1, 12)
        # bias toward N near k so the "boundary" region is well-covered,
        # but also allow larger N to stress the recurrence.
        choice = rng.random()
        if choice < 0.4:
            N = rng.randint(0, max(k + 2, 8))   # small N (hardcode-temptation zone)
        elif choice < 0.8:
            N = rng.randint(0, 200)
        else:
            N = rng.randint(0, 2000)            # larger, still brute-feasible
        # mix of prime and composite moduli, including non-prime
        p = rng.choice([2, 3, 5, 7, 13, 97, 100, 1000, 998244353, 1000000007,
                        rng.randint(2, 10**9)])
        cases.append((N, k, p))

    # Deterministic edge cases.
    edges = [
        (0, 1, 1000000007),
        (0, 5, 998244353),
        (1, 1, 7),
        (5, 1, 10**9),       # k=1: always 1
        (1, 2, 2),
        (2, 2, 5),
        (3, 2, 1000000007),  # Fibonacci-ish
        (10, 2, 1000000007),
        (10, 3, 998244353),
        (4, 4, 7),           # N == k boundary: 2^k - 1
        (5, 4, 13),
        (49, 50, 1000000007),  # N < k: pure 2^N
        (50, 50, 1000000007),  # N == k: 2^k - 1
        (51, 50, 998244353),   # N just past k
        (1000, 7, 100),        # composite modulus
        (0, 12, 2),
        (12, 12, 3),
        (200, 2, 2),
    ]
    cases.extend(edges)

    print(len(cases))
    for (N, k, p) in cases:
        print(N, k, p)


if __name__ == "__main__":
    main()
