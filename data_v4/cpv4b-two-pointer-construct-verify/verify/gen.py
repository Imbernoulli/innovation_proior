#!/usr/bin/env python3
"""Random small-case generator: python3 gen.py <seed> -> prints one n on stdout."""
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # Mix tiny edge cases with small-but-nontrivial ones.
    r = rng.random()
    if r < 0.15:
        n = 2                      # minimum
    elif r < 0.30:
        n = 3
    elif r < 0.55:
        n = rng.randint(2, 12)     # tiny (exhaustive existence path in checker)
    else:
        n = rng.randint(2, 120)    # small but past the n=4 "luck" zone
    print(n)

if __name__ == "__main__":
    main()
