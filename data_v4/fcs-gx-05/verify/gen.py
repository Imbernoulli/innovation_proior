#!/usr/bin/env python3
# Random small-case generator. Usage: gen.py <seed>
# Produces inputs in the documented format:
#   line 1: n T
#   next n lines: l_i r_i   (1 <= l_i <= r_i <= T)
import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Keep small so Kuhn's oracle is fast and feasibility is often a near thing.
    T = rng.randint(1, 8)
    # n sometimes exceeds T (forces NO), sometimes <= T.
    n = rng.randint(0, T + 2)

    print(n, T)
    out = []
    for _ in range(n):
        a = rng.randint(1, T)
        b = rng.randint(1, T)
        if a > b:
            a, b = b, a
        out.append(f"{a} {b}")
    if out:
        print("\n".join(out))

if __name__ == "__main__":
    main()
