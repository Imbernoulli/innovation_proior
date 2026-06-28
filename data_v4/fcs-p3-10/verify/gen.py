#!/usr/bin/env python3
"""Random + edge-case test generator for the prefix-sum recurrence problem.

Usage: gen.py <seed> [mode]
Prints a full stdin instance (q queries, each: a b c d N p).

Because the Python brute iterates O(N) per query, N is kept small here (so the
two solvers can be compared). The matrix solver's large-N behaviour is exercised
separately by the big-N self-check in the harness.
"""
import sys
import random


def rand_query(rng, nmax, pmax):
    p = rng.randint(1, pmax)
    a = rng.randint(-pmax, pmax)
    b = rng.randint(-pmax, pmax)
    c = rng.randint(-pmax, pmax)
    d = rng.randint(-pmax, pmax)
    N = rng.randint(0, nmax)
    return (a, b, c, d, N, p)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "rand"
    rng = random.Random(seed)

    queries = []
    if mode == "edge":
        # deterministic edge bundle
        edges = [
            (0, 0, 0, 0, 0, 1),     # p=1 -> 0
            (5, 7, 1, 1, 0, 1000),  # N=0 -> 0
            (5, 7, 1, 1, 1, 1000),  # N=1 -> a
            (5, 7, 1, 1, 2, 1000),  # N=2 -> a+b
            (-3, -9, 2, -1, 5, 7),  # negatives, small mod
            (1, 1, 1, 1, 10, 1000000007),  # classic Fibonacci prefix
            (0, 1, 1, 1, 1, 13),    # N=1 with a=0
            (1000000000, -1000000000, 999999999, -999999999, 6, 998244353),
            (2, 3, 0, 0, 7, 11),    # c=d=0 -> terms become 0 from index 2
            (2, 3, 1, 0, 7, 11),    # d=0 -> constant-ish
            (7, 7, 0, 1, 9, 100),   # only d term
        ]
        queries = edges
    else:
        # random small N so the python brute is fast
        nmax = rng.choice([3, 5, 10, 30, 100, 300])
        pmax = rng.choice([2, 3, 7, 100, 1000, 1000000007, 1999999999])
        nq = rng.randint(1, 8)
        for _ in range(nq):
            queries.append(rand_query(rng, nmax, pmax))

    out = [str(len(queries))]
    for (a, b, c, d, N, p) in queries:
        out.append(f"{a} {b} {c} {d} {N} {p}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
