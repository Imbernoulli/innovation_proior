#!/usr/bin/env python3
# Random SMALL-case generator: python3 gen.py <seed>
# Prints "n K h" with small n so the exponential brute force stays cheap.
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(1, 12)
    K = rng.randint(1, n)        # 1 <= K <= n
    h = rng.randint(0, n)        # 0 <= h <= n
    print(f"{n} {K} {h}")

if __name__ == "__main__":
    main()
