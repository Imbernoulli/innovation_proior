import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Tiny cases so brute force (O(n^2 * window)) stays cheap.
    # Mix regimes to stress the negative/zero/all-negative/empty corners.
    regime = rng.randint(0, 6)

    if regime == 0:
        n = 0  # empty
    elif regime == 1:
        n = 1
    else:
        n = rng.randint(0, 8)

    # k may exceed n (forces INFEASIBLE) or be within range.
    k = rng.randint(1, max(1, n + 2))

    if regime == 2:
        # all negative
        vals = [rng.randint(-9, -1) for _ in range(n)]
    elif regime == 3:
        # negatives and zeros only
        vals = [rng.randint(-5, 0) for _ in range(n)]
    elif regime == 4:
        # include zeros, small magnitude mixed signs
        vals = [rng.randint(-3, 3) for _ in range(n)]
    elif regime == 5:
        # all zeros
        vals = [0 for _ in range(n)]
    else:
        vals = [rng.randint(-9, 9) for _ in range(n)]

    out = [f"{n} {k}"]
    if vals:
        out.append(" ".join(str(v) for v in vals))
    else:
        out.append("")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
