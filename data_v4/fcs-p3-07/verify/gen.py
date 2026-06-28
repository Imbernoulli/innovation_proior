#!/usr/bin/env python3
"""Random + edge-case test generator for the Josephus survivor problem.

Usage: gen.py SEED [MODE]
Prints a full input file to stdout:
    q
    n k        (q lines)

Constraints honoured (matching context.md):
    1 <= n <= 1e9, 1 <= k <= 50
The brute oracle only computes the recurrence (O(n)); to keep brute fast the
generator keeps n modest in the differential-test modes. A separate 'big' mode
emits large n for performance sanity (used by the harness with the recurrence
brute only when n is moderate, or just to time sol).
"""
import random
import sys

K_MAX = 50


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "small"
    rng = random.Random(seed)

    queries = []

    if mode == "edge":
        # Deterministic edge cases.
        edges = [
            (1, 1), (1, 2), (1, 50),
            (2, 1), (2, 2), (2, 3), (2, 50),
            (3, 1), (3, 2), (3, 3),
            (5, 1), (5, 5),
            (10, 1), (10, 2), (10, 3), (10, 7),
            (100, 2), (100, 3), (100, 50),
            (1000, 2), (1000, 1), (2999, 7), (3000, 2),
        ]
        queries = edges
    elif mode == "moderate":
        # n up to a few thousand so the list-sim oracle stays exact & fast.
        m = rng.randint(50, 200)
        for _ in range(m):
            n = rng.randint(1, 3000)
            k = rng.randint(1, K_MAX)
            queries.append((n, k))
    elif mode == "midn":
        # n up to ~2e5 -> recurrence oracle still fine; sim cross-check skipped.
        m = rng.randint(20, 60)
        for _ in range(m):
            n = rng.randint(1, 200000)
            k = rng.randint(1, K_MAX)
            queries.append((n, k))
    elif mode == "tinyk":
        # focus on small k values where batching logic is most delicate
        m = rng.randint(50, 150)
        for _ in range(m):
            n = rng.randint(1, 5000)
            k = rng.randint(1, 5)
            queries.append((n, k))
    elif mode == "big":
        # Performance: large n for timing sol only.
        m = rng.randint(1, 1000)
        for _ in range(m):
            n = rng.randint(900000000, 1000000000)
            k = rng.randint(1, K_MAX)
            queries.append((n, k))
    else:  # "small"
        m = rng.randint(50, 150)
        for _ in range(m):
            n = rng.randint(1, 1000)
            k = rng.randint(1, K_MAX)
            queries.append((n, k))

    out = [str(len(queries))]
    for n, k in queries:
        out.append(f"{n} {k}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
