#!/usr/bin/env python3
# Random small-case generator for fcs-gr-03.
# Usage: gen.py <seed>
# Keeps T small (<= 8) so the W^T brute force stays tractable.
import sys
import random


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    W = rng.randint(1, 4)
    T = rng.randint(0, 7)        # exactly W^T assignments enumerated by brute

    out = []
    out.append(f"{W} {T}")
    for i in range(W):
        row = [str(rng.randint(0, 12)) for _ in range(T)]
        out.append(" ".join(row) if row else "")
    # quotas: from 0 (no free tasks) up to a bit beyond T
    q = [str(rng.randint(0, T + 1)) for _ in range(W)]
    out.append(" ".join(q))
    # overtime slopes: include 0 (no overtime) to exercise the convex edges
    base = [str(rng.randint(0, 8)) for _ in range(W)]
    out.append(" ".join(base))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
