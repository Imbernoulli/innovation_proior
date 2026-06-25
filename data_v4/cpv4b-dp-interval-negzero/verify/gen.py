import sys
import random

# Random SMALL-case generator: python3 gen.py <seed>
# Emphasizes negatives, zeros, and tiny n (including 0 and 1).

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # bias toward small n; include the empty and singleton corners frequently
    roll = rng.random()
    if roll < 0.15:
        n = rng.choice([0, 1])
    else:
        n = rng.randint(2, 8)

    # value distribution: heavy on negatives and zeros so sign handling is tested
    lo, hi = rng.choice([(-5, 5), (-9, 0), (-3, 3), (-9, 9), (0, 0)])

    vals = []
    for _ in range(n):
        r = rng.random()
        if r < 0.20:
            vals.append(0)
        else:
            vals.append(rng.randint(lo, hi))

    out = [str(n)]
    out.extend(str(v) for v in vals)
    sys.stdout.write(" ".join(out) + "\n")


if __name__ == "__main__":
    main()
