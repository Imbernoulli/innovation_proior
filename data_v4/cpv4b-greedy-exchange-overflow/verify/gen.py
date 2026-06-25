import sys
import random

# Random SMALL-case generator: python3 gen.py <seed>
# Keeps n and deadlines small so the bitmask brute force stays fast, while still
# exercising ties, equal deadlines, deadline-1 collisions, and small profits.

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(0, 8)
    # Keep maxd small (<= 10) so the brute-force bitmask is cheap.
    dmax = rng.randint(1, 8)
    lines = [str(n)]
    for _ in range(n):
        # Small profits, sometimes equal, sometimes zero, to stress tie-breaking.
        p = rng.randint(0, 12)
        d = rng.randint(1, dmax)
        lines.append(f"{p} {d}")
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
