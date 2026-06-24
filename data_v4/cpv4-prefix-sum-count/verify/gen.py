import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Small cases that still exercise the pitfalls:
    #  - small m so residue collisions (and thus pairs) are dense
    #  - negative values to stress residue normalization
    #  - occasional m = 1 (everything divisible)
    n = rng.randint(0, 12)
    m = rng.choice([1, 1, 2, 3, 4, 5, 6, 7])
    # value range chosen so windows often hit a multiple of m
    lo, hi = -8, 8
    a = [rng.randint(lo, hi) for _ in range(n)]

    out = []
    out.append(f"{n} {m}")
    out.append(" ".join(map(str, a)))
    sys.stdout.write("\n".join(out) + "\n")

main()
