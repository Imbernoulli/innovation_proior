import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes to stress negatives, zeros, all-negative, empty.
    regime = rng.randint(0, 6)
    if regime == 0:
        n = 0
    elif regime == 1:
        n = 1
    else:
        n = rng.randint(1, 8)

    if regime == 2:
        # all negative
        lo, hi = -9, -1
    elif regime == 3:
        # negatives and zeros only
        lo, hi = -6, 0
    elif regime == 4:
        # include positives, negatives, zeros, small range
        lo, hi = -4, 4
    elif regime == 5:
        # mostly zeros
        lo, hi = -2, 2
    else:
        lo, hi = -9, 9

    a = [rng.randint(lo, hi) for _ in range(n)]
    out = [str(n)]
    if n:
        out.append(" ".join(map(str, a)))
    print("\n".join(out))

if __name__ == "__main__":
    main()
