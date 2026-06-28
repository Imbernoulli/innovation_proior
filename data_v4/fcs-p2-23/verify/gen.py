#!/usr/bin/env python3
"""Random + structured test generator for the Stone Game problem.

Usage: gen.py SEED [MODE]
Prints a valid stdin instance to stdout.

Modes bias toward small n (so the DP and an exhaustive minimax agree cheaply),
plus value patterns that stress the greedy 'take larger end' heuristic.
"""
import random
import sys


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    mode = sys.argv[2] if len(sys.argv) > 2 else "rand"
    rng = random.Random(seed)

    if mode == "tiny":
        n = rng.randint(0, 6)
        a = [rng.randint(-9, 9) for _ in range(n)]
    elif mode == "small":
        n = rng.randint(1, 12)
        a = [rng.randint(-20, 20) for _ in range(n)]
    elif mode == "smallpos":
        n = rng.randint(1, 12)
        a = [rng.randint(0, 50) for _ in range(n)]
    elif mode == "greedytrap":
        # alternating big/small so 'take larger end' misleads
        n = rng.randint(2, 14)
        a = []
        for i in range(n):
            a.append(rng.randint(80, 100) if i % 2 == 0 else rng.randint(0, 5))
        rng.shuffle(a) if rng.random() < 0.3 else None
    elif mode == "mid":
        n = rng.randint(1, 60)
        a = [rng.randint(-1000, 1000) for _ in range(n)]
    elif mode == "big":
        n = rng.randint(1, 2000)
        a = [rng.randint(-10**9, 10**9) for _ in range(n)]
    elif mode == "edge0":
        n = 0
        a = []
    elif mode == "edge1":
        n = 1
        a = [rng.choice([-10**9, -1, 0, 1, 10**9, rng.randint(-1000, 1000)])]
    elif mode == "extreme":
        # maximal magnitude to probe overflow / parity
        n = rng.randint(1, 2000)
        a = [rng.choice([-10**9, 10**9]) for _ in range(n)]
    else:  # rand
        n = rng.randint(0, 14)
        a = [rng.randint(-30, 30) for _ in range(n)]

    out = [str(n)]
    out.append(" ".join(map(str, a)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
