#!/usr/bin/env python3
"""
Random + edge-case generator for the house-painting problem.

Usage: gen.py <seed> [mode]

Emits a single test case on stdout in the format:
    n k
    cost[0][0..k-1]
    ...
    cost[n-1][0..k-1]

Modes bias toward different stress patterns; default picks one at random based
on the seed. Costs include large values to exercise 64-bit accumulation, plus
adversarial patterns where one color is cheap everywhere (the greedy trap).
"""
import random
import sys


def emit(n, k, costs):
    out = [f"{n} {k}"]
    for row in costs:
        out.append(" ".join(str(x) for x in row))
    sys.stdout.write("\n".join(out) + "\n")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else None
    rng = random.Random(seed)
    if mode is None:
        mode = rng.choice([
            "tiny", "tiny", "tiny",      # weight tiny for exhaustive cross-check
            "small", "small",
            "greedy_trap", "greedy_trap",
            "two_color", "k1", "big", "uniform", "medium",
        ])

    if mode == "tiny":
        n = rng.randint(0, 6)
        k = rng.randint(1, 4)
        costs = [[rng.randint(0, 9) for _ in range(k)] for _ in range(n)]
        emit(n, k, costs)

    elif mode == "small":
        n = rng.randint(1, 8)
        k = rng.randint(2, 5)
        costs = [[rng.randint(0, 20) for _ in range(k)] for _ in range(n)]
        emit(n, k, costs)

    elif mode == "k1":
        # single color: impossible for n>=2, trivial for n<=1
        n = rng.randint(0, 6)
        costs = [[rng.randint(0, 50)] for _ in range(n)]
        emit(n, 1, costs)

    elif mode == "two_color":
        n = rng.randint(1, 8)
        k = 2
        costs = [[rng.randint(0, 30) for _ in range(2)] for _ in range(n)]
        emit(n, 2, costs)

    elif mode == "greedy_trap":
        # One "globally cheapest" color that, if always taken, forces expensive
        # neighbours. Classic case where cheapest-non-conflicting greedy loses.
        n = rng.randint(2, 8)
        k = rng.randint(2, 4)
        cheap = rng.randrange(k)
        costs = []
        for _ in range(n):
            row = [rng.randint(40, 100) for _ in range(k)]
            row[cheap] = rng.randint(0, 5)
            costs.append(row)
        emit(n, k, costs)

    elif mode == "uniform":
        n = rng.randint(1, 8)
        k = rng.randint(2, 4)
        v = rng.randint(0, 1000)
        costs = [[v for _ in range(k)] for _ in range(n)]
        emit(n, k, costs)

    elif mode == "medium":
        n = rng.randint(1, 40)
        k = rng.randint(2, 8)
        costs = [[rng.randint(0, 1000) for _ in range(k)] for _ in range(n)]
        emit(n, k, costs)

    elif mode == "big":
        # Large values to exercise 64-bit sums (still small n so brute is fast).
        n = rng.randint(1, 30)
        k = rng.randint(2, 6)
        costs = [[rng.randint(900000000, 1000000000) for _ in range(k)]
                 for _ in range(n)]
        emit(n, k, costs)

    else:
        emit(1, 1, [[0]])


if __name__ == "__main__":
    main()
