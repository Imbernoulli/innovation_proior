#!/usr/bin/env python3
# Random + edge-case generator for "Maximum-sum strictly increasing subsequence".
# Usage: gen.py SEED  -> prints one test case to stdout.
# Keeps n small so the exhaustive brute oracle stays feasible, and stresses the
# patterns that break greedy: many equal values, decreasing runs, negatives,
# and "small-then-one-big" traps.
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    mode = seed % 11
    if mode == 0:
        n = 0
        a = []
    elif mode == 1:
        n = 1
        a = [rng.randint(-20, 20)]
    elif mode == 2:
        # all equal -> only length-1 subsequences are strictly increasing
        n = rng.randint(1, 14)
        v = rng.randint(-10, 10)
        a = [v] * n
    elif mode == 3:
        # strictly decreasing -> answer is the single largest element (or 0)
        n = rng.randint(1, 14)
        a = sorted([rng.randint(-15, 15) for _ in range(n)], reverse=True)
    elif mode == 4:
        # all negative -> answer should be 0 (empty)
        n = rng.randint(1, 14)
        a = [rng.randint(-30, -1) for _ in range(n)]
    elif mode == 5:
        # tiny values with many duplicates -> greedy traps
        n = rng.randint(2, 16)
        a = [rng.randint(-3, 3) for _ in range(n)]
    elif mode == 6:
        # "greedy trap": small increasing run, then a single huge value
        n = rng.randint(3, 12)
        a = [rng.randint(1, 4) for _ in range(n - 1)] + [rng.randint(50, 100)]
        rng.shuffle(a)
    elif mode == 7:
        # increasing then big negatives interleaved
        n = rng.randint(2, 16)
        a = [rng.randint(-25, 25) for _ in range(n)]
    elif mode == 8:
        # mostly sorted ascending with some noise
        n = rng.randint(2, 16)
        a = sorted([rng.randint(-15, 15) for _ in range(n)])
        for _ in range(rng.randint(0, n)):
            i = rng.randrange(n)
            a[i] = rng.randint(-15, 15)
    else:
        # generic random
        n = rng.randint(0, 16)
        a = [rng.randint(-40, 40) for _ in range(n)]

    out = [str(n)]
    if a:
        out.append(" ".join(map(str, a)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
