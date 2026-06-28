#!/usr/bin/env python3
# Random small-case generator. Usage: gen.py <seed>
# Emits a single integer n in a range where the O(n log n) brute force is fast.
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # Mostly small n so brute is cheap; occasionally hit small edge values.
    r = rng.random()
    if r < 0.15:
        n = rng.randint(0, 5)          # tiny / edge region (incl. 0, 1)
    elif r < 0.30:
        n = rng.randint(0, 50)
    else:
        n = rng.randint(1, 20000)      # main band, still fast for the sieve
    print(n)

if __name__ == "__main__":
    main()
