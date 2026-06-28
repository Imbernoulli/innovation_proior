#!/usr/bin/env python3
"""Random + edge-case generator for the matrix-chain problem.

Usage: gen.py <seed> [mode]
Prints a valid test case to stdout: n on the first line, then n+1 dims.
Modes bias toward small n (so the brute oracle stays cheap) and toward
configurations that stress greedy ('multiply cheapest adjacent pair first').
"""
import sys
import random


def emit(n, p):
    print(n)
    print(" ".join(str(x) for x in p))


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "rand"
    rng = random.Random(seed)

    if mode == "edge":
        # A small fixed library of corner cases, indexed by seed.
        cases = [
            (0, [1]),                       # zero matrices
            (1, [3, 4]),                    # single matrix
            (2, [2, 3, 4]),                 # one multiplication
            (2, [1, 1, 1]),                 # trivial dims
            (3, [10, 1, 10, 10]),           # classic greedy trap
            (3, [40, 20, 30, 10]),          # textbook ordering matters
            (4, [5, 4, 6, 2, 7]),           # classic CLRS example
            (5, [30, 35, 15, 5, 10, 20]),   # classic 5-matrix chain (opt 11875)
            (2, [1000, 1000, 1000]),        # max dims, tiny n
            (3, [1, 1000, 1, 1000]),        # extreme spread
        ]
        idx = seed % len(cases)
        n, p = cases[idx]
        emit(n, p)
        return

    if mode == "trap":
        # Build chains designed to fool 'cheapest adjacent pair first':
        # alternate tiny and huge dimensions so the locally cheapest merge
        # is globally suboptimal.
        n = rng.randint(3, 9)
        p = []
        for _ in range(n + 1):
            if rng.random() < 0.5:
                p.append(rng.randint(1, 3))
            else:
                p.append(rng.randint(80, 1000))
        emit(n, p)
        return

    # Default: small random chains so the enumeration brute terminates fast.
    n = rng.randint(0, 11)
    hi = rng.choice([5, 20, 100, 1000])
    p = [rng.randint(1, hi) for _ in range(n + 1)]
    emit(n, p)


if __name__ == "__main__":
    main()
