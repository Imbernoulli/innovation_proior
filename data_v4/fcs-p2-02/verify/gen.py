#!/usr/bin/env python3
"""Random + edge-case generator for weighted interval scheduling.

Usage: gen.py SEED [MODE]
Prints a test case to stdout in the judge format:
    n
    s e w        (n lines; half-open interval [s, e) with s < e, weight w >= 1)
"""
import random
import sys


def emit(jobs):
    out = [str(len(jobs))]
    for (s, e, w) in jobs:
        out.append(f"{s} {e} {w}")
    sys.stdout.write("\n".join(out) + "\n")


def rand_interval(coord_max, w_max, rng):
    s = rng.randint(0, coord_max)
    e = rng.randint(s + 1, s + 1 + coord_max)  # ensure s < e
    w = rng.randint(1, w_max)
    return (s, e, w)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "small"
    rng = random.Random(seed)

    if mode == "edge":
        # Pick one of a handful of hand-built corner cases, indexed by seed.
        cases = []
        cases.append([])                                   # n = 0
        cases.append([(0, 1, 1)])                          # n = 1
        cases.append([(0, 5, 10), (5, 10, 10)])            # touching, both fit
        cases.append([(0, 5, 10), (4, 10, 100)])           # overlap, take heavier
        cases.append([(0, 10, 1), (0, 10, 1), (0, 10, 1)]) # identical, only one
        cases.append([(0, 3, 5), (1, 4, 4), (3, 6, 5)])    # classic chain vs middle
        # Earliest-finish-time greedy trap: a short cheap interval finishing first
        # blocks one long valuable interval.
        cases.append([(0, 1, 1), (0, 100, 1000)])
        # Greedy (unweighted activity-selection) trap with more items.
        cases.append([(0, 2, 1), (1, 3, 1), (2, 4, 1), (0, 4, 5)])
        # Nested intervals.
        cases.append([(0, 100, 50), (10, 20, 30), (30, 40, 30), (50, 60, 30)])
        # All sharing one point.
        cases.append([(0, 5, 7), (3, 8, 7), (6, 9, 7)])
        idx = seed % len(cases)
        emit(cases[idx])
        return

    if mode == "small":
        # n up to 18 so the exhaustive oracle is feasible; small coords -> many overlaps.
        n = rng.randint(0, 12)
        coord_max = rng.choice([3, 5, 8, 12])
        w_max = rng.choice([1, 3, 10, 1000])
        jobs = [rand_interval(coord_max, w_max, rng) for _ in range(n)]
        emit(jobs)
        return

    if mode == "mid":
        # n in a range where the O(n^2) oracle is still fast (<= ~2000).
        n = rng.randint(19, 400)
        coord_max = rng.choice([20, 100, 1000, 10**9])
        w_max = rng.choice([1, 100, 10**9])
        jobs = [rand_interval(coord_max, w_max, rng) for _ in range(n)]
        emit(jobs)
        return

    if mode == "big":
        # Stress the main solution's performance / overflow (compare not required).
        n = 200000
        coord_max = 10**9
        jobs = []
        for _ in range(n):
            s = rng.randint(0, coord_max)
            e = rng.randint(s + 1, min(s + 1 + 1000, 2 * 10**9))
            w = rng.randint(1, 10**9)
            jobs.append((s, e, w))
        emit(jobs)
        return

    raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
