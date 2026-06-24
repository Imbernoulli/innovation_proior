import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes to exercise the sign/zero/empty/all-negative corners.
    r = rng.random()
    if r < 0.10:
        n = 0                      # empty array corner
    elif r < 0.25:
        n = 1                      # single element (often a lone negative / zero)
    else:
        n = rng.randint(2, 14)     # small so the O(n^2) brute is exact and fast

    # Values bounded to [-2, 2] so products stay within 64-bit even at n=62.
    # Bias toward producing negatives and zeros frequently.
    vals = []
    for _ in range(n):
        choice = rng.random()
        if choice < 0.18:
            vals.append(0)
        elif choice < 0.59:
            vals.append(rng.randint(-2, -1))   # negatives common
        else:
            vals.append(rng.randint(1, 2))

    # Occasionally force an all-negative array to stress the sign base case.
    if n >= 1 and rng.random() < 0.20:
        vals = [rng.randint(-2, -1) for _ in range(n)]

    out = [str(n)]
    out.extend(str(v) for v in vals)
    sys.stdout.write(" ".join(out) + "\n")

main()
