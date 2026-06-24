import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes to stress base case / sign handling:
    #  - sometimes n = 0 (empty)
    #  - sometimes all-negative arrays
    #  - small magnitudes so brute is exact and fast
    mode = rng.randint(0, 4)
    if mode == 0:
        n = 0
    else:
        n = rng.randint(1, 8)

    if mode == 1:
        # all negative values
        a = [rng.randint(-6, -1) for _ in range(n)]
    elif mode == 2:
        # negatives and zeros only
        a = [rng.randint(-6, 0) for _ in range(n)]
    else:
        a = [rng.randint(-6, 6) for _ in range(n)]

    # cost c is non-negative; sometimes large enough to make every block unprofitable
    cmode = rng.randint(0, 3)
    if cmode == 0:
        c = 0
    elif cmode == 1:
        c = rng.randint(0, 3)
    else:
        c = rng.randint(0, 30)

    out = [f"{n} {c}"]
    if n > 0:
        out.append(" ".join(str(x) for x in a))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
