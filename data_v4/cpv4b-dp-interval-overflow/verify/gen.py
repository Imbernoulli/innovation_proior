import sys, random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # Small cases so the exponential brute force stays fast.
    # Mix of edge sizes (0,1) and small n.
    r = rng.random()
    if r < 0.08:
        n = 0
    elif r < 0.18:
        n = 1
    else:
        n = rng.randint(2, 8)
    # Occasionally use larger weights to exercise wider sums (still safe for brute).
    hi = rng.choice([5, 20, 1000, 100000])
    w = [rng.randint(0, hi) for _ in range(n)]
    out = [str(n)]
    if w:
        out.append(" ".join(map(str, w)))
    print("\n".join(out))

if __name__ == "__main__":
    main()
