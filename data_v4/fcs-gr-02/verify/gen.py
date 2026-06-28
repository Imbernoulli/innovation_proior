#!/usr/bin/env python3
# Random small-case generator for differential testing.
# Usage: gen.py <seed>
# Emits a 2-SAT instance: n variables, m clauses, each clause "(i a) OR (j b)".
import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # small n so brute force (2^n) is cheap
    n = rng.randint(1, 8)
    # bias toward instances that are "tight" (more likely UNSAT) sometimes
    mode = rng.randint(0, 3)
    if mode == 0:
        m = rng.randint(0, 3)                 # few clauses -> usually SAT
    elif mode == 3:
        m = rng.randint(n, 4 * n + 6)         # many clauses -> often UNSAT
    else:
        m = rng.randint(0, 2 * n + 3)

    lines = [f"{n} {m}"]
    for _ in range(m):
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        a = rng.randint(0, 1)
        b = rng.randint(0, 1)
        lines.append(f"{i} {a} {j} {b}")
    sys.stdout.write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
