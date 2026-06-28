#!/usr/bin/env python3
"""Random + edge-case test generator for the lattice-path problem.

Usage: gen.py <seed> [mode]

Because the brute oracle (brute.py) builds an O(N^2) Pascal table, the random
tests keep a+b SMALL so the differential test runs fast. The trap of the
problem (hardcoding small cases) is precisely tested by these small inputs:
the multiplicative C(a+b,a) solution must agree with the additive Pascal
table on every small grid, and the editorial argues the same code scales to
a+b up to 2*10^6 where any hardcoded table would be impossible.

Modes (chosen by seed % k when mode not given):
  random   - random small a,b
  edges    - include 0-coordinate corners, squares, single steps
  tiny     - very small grids (the "hardcodable looking" regime)
"""
import random
import sys

MOD = 1000000007


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else None
    rng = random.Random(seed)

    if mode is None:
        mode = rng.choice(["random", "random", "edges", "tiny"])

    queries = []

    if mode == "edges":
        # Deterministic-ish edge bag plus a few randoms.
        bag = [
            (0, 0),
            (0, 1), (1, 0),
            (0, 5), (5, 0),
            (1, 1),
            (2, 2),
            (5, 5),
            (10, 0), (0, 10),
            (7, 3), (3, 7),
            (1, 9), (9, 1),
        ]
        # keep a+b small for the brute
        bag = [(a, b) for (a, b) in bag if a + b <= 60]
        rng.shuffle(bag)
        k = rng.randint(1, len(bag))
        queries = bag[:k]
    elif mode == "tiny":
        # Very small grids that look hardcodable.
        q = rng.randint(1, 12)
        for _ in range(q):
            a = rng.randint(0, 6)
            b = rng.randint(0, 6)
            queries.append((a, b))
    else:  # random
        q = rng.randint(1, 10)
        cap = rng.choice([20, 40, 60, 80])
        for _ in range(q):
            a = rng.randint(0, cap)
            b = rng.randint(0, cap - a) if cap - a > 0 else 0
            # ensure a+b <= cap to keep brute table small
            queries.append((a, b))

    if not queries:
        queries = [(0, 0)]

    lines = [str(len(queries))]
    for a, b in queries:
        lines.append(f"{a} {b}")
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
