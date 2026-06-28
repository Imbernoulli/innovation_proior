#!/usr/bin/env python3
import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # Small cases so brute force (range scan) is feasible: keep R small.
    mode = rng.randint(0, 4)
    if mode == 0:
        # tiny range
        L = rng.randint(1, 50)
        R = L + rng.randint(0, 30)
    elif mode == 1:
        # spanning a power-of-ten boundary (length change), small magnitude
        base = 10 ** rng.randint(1, 4)
        L = max(1, base - rng.randint(0, 20))
        R = base + rng.randint(0, 20)
    elif mode == 2:
        # moderate range up to a few thousand
        L = rng.randint(1, 2000)
        R = L + rng.randint(0, 4000)
    elif mode == 3:
        # single point
        L = rng.randint(1, 100000)
        R = L
    else:
        # bigger but still brute-feasible window
        L = rng.randint(1, 500000)
        span = rng.randint(0, 100000)
        R = L + span
    print(L, R)

if __name__ == "__main__":
    main()
