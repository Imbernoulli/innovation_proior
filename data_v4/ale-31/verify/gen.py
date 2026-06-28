#!/usr/bin/env python3
"""Instance generator for "Balanced Districting" (graph partition, ALE-Bench).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout:

    H W K
    p_{0,0} p_{0,1} ... p_{0,W-1}
    p_{1,0} ...
    ...
    p_{H-1,0} ... p_{H-1,W-1}

`H` rows and `W` columns of an integer POPULATION grid (each cell has a weight
p >= 1), and `K` the number of districts to form. Every cell must end up in
exactly one of K districts; each district must be a single 4-connected region.

The populations are NOT uniform: they are a smooth low-frequency "population
density" field (a sum of a few 2-D Gaussian bumps over a flat background) sampled
on the grid and quantised to small positive integers, plus a little per-cell
noise. This is what makes balancing non-trivial -- a naive stripe partition cuts
straight through dense bumps and ends up wildly imbalanced, while a good
partition has to bend its district boundaries around the population hotspots.
The boundary-length penalty then fights back against districts that get too
fractal in chasing balance, so the optimum is a genuine trade-off.

H, W, K are chosen deterministically from the seed.
"""
import sys
import math
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x31A1_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    # grid size and number of districts, deterministic from the seed
    H = rng.randint(20, 40)
    W = rng.randint(20, 40)
    K = rng.randint(4, 10)
    # never ask for more districts than cells (always satisfiable)
    K = min(K, H * W)

    # smooth population field: a flat background + a few Gaussian "hot spots"
    num_bumps = rng.randint(3, 7)
    bumps = []
    for _ in range(num_bumps):
        cx = rng.uniform(0, W - 1)
        cy = rng.uniform(0, H - 1)
        amp = rng.uniform(6.0, 20.0)
        sig = rng.uniform(0.10, 0.30) * max(H, W)
        bumps.append((cx, cy, amp, sig))

    base = rng.uniform(1.0, 3.0)

    grid = [[0] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            v = base
            for (cx, cy, amp, sig) in bumps:
                dx = c - cx
                dy = r - cy
                v += amp * math.exp(-(dx * dx + dy * dy) / (2.0 * sig * sig))
            # a little multiplicative noise so cells aren't perfectly smooth
            v *= rng.uniform(0.85, 1.15)
            p = int(round(v))
            if p < 1:
                p = 1  # populations are strictly positive
            grid[r][c] = p

    out = [f"{H} {W} {K}"]
    for r in range(H):
        out.append(" ".join(str(grid[r][c]) for c in range(W)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
