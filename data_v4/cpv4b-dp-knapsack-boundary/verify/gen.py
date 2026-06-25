import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases so the 2^n brute force is feasible and the price-window boundary
    # gets exercised: small n, small prices, R near the sum of prices.
    n = rng.randint(0, 12)
    p = [rng.randint(1, 8) for _ in range(n)]
    v = [rng.randint(-6, 6) for _ in range(n)]

    total = sum(p)
    # Pick a window [L, R] with R chosen anywhere from 0 to a bit above total,
    # and L from 0 to R, so empty/IMPOSSIBLE/exact-edge cases all occur.
    hi = max(total, 1)
    R = rng.randint(0, hi + 2)
    L = rng.randint(0, R)

    out = []
    out.append(f"{n} {L} {R}")
    for i in range(n):
        out.append(f"{p[i]} {v[i]}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
