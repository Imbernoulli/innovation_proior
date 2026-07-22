#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of the cospectral-decoy-mints problem.

Instance = "n M" where n is the number of vertices of both coins (graphs) G and H
(always a multiple of 6), and M is the per-graph edge budget. Difficulty ladder is a
fixed, deterministic function of testId (no external randomness needed at all).
"""
import sys

# k = number of 6-vertex "gadget blocks" available at each difficulty level.
K_LADDER = [2, 3, 4, 5, 6, 6, 7, 7, 8, 8]


def main():
    test_id = int(sys.argv[1])
    idx = max(1, min(test_id, len(K_LADDER))) - 1
    k = K_LADDER[idx]
    n = 6 * k
    m = 2 * n
    print(n, m)


if __name__ == "__main__":
    main()
