import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases so the brute force stays fast.
    r = rng.random()
    if r < 0.06:
        n = 1
    elif r < 0.12:
        n = 2
    else:
        n = rng.randint(3, 9)

    # L: the per-rug inclusive length cap. Push the boundary by often making
    # L exactly 1, or exactly n, or near n, to stress the inclusive length
    # constraint l..r with length r-l+1 <= L.
    Lchoice = rng.random()
    if Lchoice < 0.25:
        L = 1
    elif Lchoice < 0.45:
        L = n
    elif Lchoice < 0.6:
        L = max(1, n - 1)
    else:
        L = rng.randint(1, max(1, n))

    # K: per-rug fixed cost. Mix small and large so that "fewer rugs" vs
    # "smaller max" trade off in both directions (greedy traps).
    K = rng.choice([0, 1, 2, 5, 10, 50, 1000])

    hi = rng.choice([1, 3, 5, 20, 100, 1000])
    a = [rng.randint(1, hi) for _ in range(n)]

    out = [f"{n} {K} {L}"]
    out.append(" ".join(map(str, a)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
