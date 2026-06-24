import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)
    # Small n so the brute force (n! permutations) stays fast.
    n = rng.randint(1, 7)
    # Mix of small and large values to exercise overflow-ish sums and ties.
    print(n)
    rows = []
    for i in range(n):
        row = []
        for j in range(n):
            # Occasionally use very large values to push sums past 2^31.
            r = rng.random()
            if r < 0.3:
                v = rng.randint(0, 10**9)
            elif r < 0.6:
                v = rng.randint(900_000_000, 10**9)
            else:
                v = rng.randint(0, 1000)
            row.append(v)
        rows.append(" ".join(map(str, row)))
    print("\n".join(rows))

main()
