#!/usr/bin/env python3
"""Random + edge-case generator for the egg-drop problem.

Usage: gen.py <seed>
Prints one test case "k m" to stdout.

m is kept reasonably small so the O(k*m^2) brute oracle stays fast, while still
exercising the regimes that matter: 1 egg (linear), 2 eggs (the classic triangular
case), many eggs (binary-search regime), and the boundary m around powers of two.
"""
import random
import sys

# Curated edge cases (returned for small seeds), then random.
EDGE = [
    (1, 1), (1, 2), (1, 10), (1, 200), (1, 500),
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 10), (2, 100), (2, 300), (2, 500),
    (3, 1), (3, 7), (3, 8), (3, 25), (3, 500),
    (4, 14), (4, 15), (4, 500),
    (5, 31), (5, 32),
    (10, 1), (10, 500), (10, 1023), (10, 1024),
    (50, 1), (50, 2), (50, 500),
    (100, 1), (100, 2), (100, 7), (100, 500),
    (1, 4), (1, 8), (1, 16),
    (2, 15), (2, 21), (2, 28),  # triangular numbers k(k+1)/2 boundaries
    (3, 14), (3, 15), (3, 16),
    (7, 127), (7, 128),
]


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if seed < len(EDGE):
        k, m = EDGE[seed]
        print(k, m)
        return
    rnd = random.Random(seed)
    # bias toward small egg counts (where the limited-egg effect is strong)
    r = rnd.random()
    if r < 0.30:
        k = rnd.choice([1, 2])
    elif r < 0.55:
        k = rnd.randint(1, 6)
    elif r < 0.80:
        k = rnd.randint(1, 20)
    else:
        k = rnd.randint(1, 100)
    # keep m small enough for the O(k*m^2) brute; emphasise small/medium m.
    rm = rnd.random()
    if rm < 0.4:
        m = rnd.randint(1, 40)
    elif rm < 0.8:
        m = rnd.randint(1, 200)
    else:
        m = rnd.randint(1, 600)
    print(k, m)


if __name__ == "__main__":
    main()
