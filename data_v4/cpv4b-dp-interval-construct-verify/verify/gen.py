#!/usr/bin/env python3
"""Random small-case generator: python3 gen.py <seed> -> a single integer n on stdout.

Keep n small so the backtracking brute terminates quickly. n in [1, 14].
"""
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(1, 14)
    print(n)

if __name__ == "__main__":
    main()
