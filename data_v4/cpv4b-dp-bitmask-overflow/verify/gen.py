import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # small n for brute-force comparison (permutations are n!)
    n = rng.randint(0, 7)

    # Mix of value regimes so that occasionally the running sum is large
    # enough to overflow a 32-bit accumulator, exercising the int64 path.
    regime = rng.randint(0, 2)
    if regime == 0:
        lo, hi = 0, 9
    elif regime == 1:
        lo, hi = 0, 1000
    else:
        lo, hi = 900_000_000, 1_000_000_000  # large: row sum can exceed 2^31

    print(n)
    for i in range(n):
        row = [str(rng.randint(lo, hi)) for _ in range(n)]
        print(" ".join(row))

if __name__ == "__main__":
    main()
