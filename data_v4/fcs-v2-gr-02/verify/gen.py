#!/usr/bin/env python3
import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    n = rng.randint(1, 9)
    # allow self-loops and multi-edges; allow some unreachable nodes
    max_m = n * n
    m = rng.randint(0, max_m)
    s = rng.randint(1, n)

    edges = []
    for _ in range(m):
        a = rng.randint(1, n)
        b = rng.randint(1, n)
        edges.append((a, b))

    out = [f"{n} {m} {s}"]
    for a, b in edges:
        out.append(f"{a} {b}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
