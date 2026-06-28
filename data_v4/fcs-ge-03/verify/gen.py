#!/usr/bin/env python3
# Random small-case generator. Usage: gen.py <seed>
# Emits n points with small coordinate range to force frequent collisions
# and near-duplicates (the hard cases for closest pair).
import sys
import random

def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    rng = random.Random(seed)

    # Mix of regimes by seed parity to exercise different structures.
    mode = seed % 4
    if mode == 0:
        n = rng.randint(0, 12)
        lo, hi = -5, 5            # tiny range -> many duplicates / ties
    elif mode == 1:
        n = rng.randint(2, 30)
        lo, hi = -20, 20
    elif mode == 2:
        n = rng.randint(2, 60)
        lo, hi = -3, 3            # extremely dense, forces duplicate points
    else:
        n = rng.randint(2, 50)
        lo, hi = -1000000000, 1000000000  # large coords -> overflow stress

    pts = []
    for _ in range(n):
        x = rng.randint(lo, hi)
        y = rng.randint(lo, hi)
        pts.append((x, y))

    out = [str(n)]
    for x, y in pts:
        out.append(f"{x} {y}")
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
