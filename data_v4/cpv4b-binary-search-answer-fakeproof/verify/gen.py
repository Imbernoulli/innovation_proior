import sys, random

# Random SMALL-case generator: python3 gen.py <seed>
# Keeps K small so the brute force (which scans m upward) stays fast,
# while still exercising the binary search + hyperbola predicate.
def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    q = rng.randint(1, 6)
    print(q)
    for _ in range(q):
        # Mix of regimes: tiny K (edge), and moderate K.
        r = rng.random()
        if r < 0.25:
            K = rng.randint(1, 5)            # tiny / edge
        elif r < 0.7:
            K = rng.randint(1, 300)          # small
        else:
            K = rng.randint(1, 3000)         # moderate (still brute-fast)
        print(K)

if __name__ == "__main__":
    main()
