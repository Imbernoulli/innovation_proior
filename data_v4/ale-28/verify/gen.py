#!/usr/bin/env python3
"""Instance generator for "Sensor Placement for Coverage + Connectivity"
(ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the format:

    H W k r lam
    d[0][0] d[0][1] ... d[0][W-1]
    d[1][0] ...
    ...
    d[H-1][0] ... d[H-1][W-1]

Meaning: an H x W grid of cells. Cell (i, j) (row i, column j, 0-based) has a
non-negative integer demand d[i][j]. We may place at most k sensors on distinct
cells. A sensor at cell (i, j) is a disk of radius r centred at the cell centre;
it COVERS every cell (a, b) whose centre lies within Euclidean distance r, i.e.
(a-i)^2 + (b-j)^2 <= r^2. The covered demand is the sum of d over the UNION of
the covered cells. Two placed sensors are "linked" if their centres are within
distance 2r (their disks touch or overlap), i.e. (i1-i2)^2 + (j1-j2)^2 <= (2r)^2;
the sensors form a connectivity graph and C is its number of connected
components. The objective (MAXIMIZE) is

    objective = covered_demand - lam * (C - 1)

so an extra disconnected cluster is penalised by lam. See score.py / context.md.

Instance regime (deterministic from the seed):
  * Grid H, W in roughly [24, 40].
  * Demand is a mixture of a few 2-D Gaussian "hotspots" (clustered demand) plus
    a low uniform background, integer-valued. Clustering is what makes coverage
    placement non-trivial AND puts the hotspots far enough apart that a pure
    coverage-greedy tends to leave the chosen sensors in disconnected clusters --
    exactly the regime where the connectivity penalty bites and a coverage +
    connectivity-repair heuristic earns its keep.
  * r is a small radius (2..4); k is sized so that sensors cover a good fraction
    of, but not all of, the high-demand area.
  * lam (the connectivity penalty per extra component) is chosen on the scale of
    a typical hotspot's coverable demand, so connecting clusters genuinely trades
    against grabbing one more hotspot.
"""
import sys
import math
import random


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x5E_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    H = rng.randint(26, 40)
    W = rng.randint(26, 40)
    r = rng.randint(2, 4)
    # number of hotspots
    nhot = rng.randint(3, 6)
    # k sized to cover a chunk but not all of the high-demand area
    k = rng.randint(8, 16)
    lam = rng.randint(40, 120)

    # Background low demand.
    d = [[rng.randint(0, 2) for _ in range(W)] for _ in range(H)]

    # Add Gaussian hotspots, placed so clusters are spread out (encourages the
    # greedy coverage to split into disconnected clusters).
    centers = []
    for _ in range(nhot):
        ci = rng.uniform(2, H - 3)
        cj = rng.uniform(2, W - 3)
        centers.append((ci, cj))
        amp = rng.uniform(8.0, 22.0)
        sig = rng.uniform(1.6, 3.2)
        inv2s2 = 1.0 / (2.0 * sig * sig)
        for i in range(H):
            for j in range(W):
                dd = (i - ci) ** 2 + (j - cj) ** 2
                add = amp * math.exp(-dd * inv2s2)
                d[i][j] += int(round(add))

    # clamp demand to a sane non-negative integer range
    for i in range(H):
        for j in range(W):
            if d[i][j] < 0:
                d[i][j] = 0
            if d[i][j] > 99:
                d[i][j] = 99

    out = [f"{H} {W} {k} {r} {lam}"]
    for i in range(H):
        out.append(" ".join(str(d[i][j]) for j in range(W)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
