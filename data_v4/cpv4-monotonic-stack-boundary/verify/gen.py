import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    n = rng.randint(0, 9)
    # Small value range so ties are frequent -> exercises the strict/non-strict
    # boundary split. Occasionally allow negatives to exercise the mod folding.
    if rng.random() < 0.3:
        lo, hi = -4, 4
    else:
        lo, hi = 1, 4
    a = [rng.randint(lo, hi) for _ in range(n)]
    out = [str(n)] + [str(x) for x in a]
    print(" ".join(out))

if __name__ == "__main__":
    main()
