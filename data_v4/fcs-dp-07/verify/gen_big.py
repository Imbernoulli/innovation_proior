#!/usr/bin/env python3
# Generate large-range cases (cross-checked against brute_big.py, not the scan brute).
import sys, random
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    MAX = 10**18
    a = rng.randint(1, MAX)
    b = rng.randint(1, MAX)
    L, R = min(a, b), max(a, b)
    print(L, R)
if __name__ == "__main__":
    main()
