#!/usr/bin/env python3
"""Instance generator for "Interactive Adaptive Probing" (ale-42).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the format:

    H W Q rad sigma thr penalty
    r_0_0 r_0_1 ... r_0_{W-1}
    r_1_0 ...
    ...
    r_{H-1}_0 ... r_{H-1}_{W-1}

Meaning. A hidden H x W grid carries a non-negative integer REWARD field
r(x, y) (row x in [0, H), column y in [0, W)). The field is a sum of a few
smooth 2-D Gaussian "hotspots" sitting on a small uniform background, scaled to
integers. A cell is HOT iff r(x, y) >= thr.

The solver is a probing agent with a budget of Q probes. A probe at cell (x, y)
"observes" every cell within Chebyshev radius `rad` of (x, y) (a (2 rad + 1)^2
window, clipped to the grid). A real probe returns only a NOISY local reading
of the window (measurement noise has std `sigma`, relative to the reward scale),
so the agent never sees the true field directly -- it must decide, from noisy
readings, which observed cells are actually hot. The final deliverable is a
REPORT: a set of cells the agent declares hot.

Scoring (see score.py / context.md for the exact rule and the feasibility -> 0
floor). A reported cell must have been OBSERVED (within `rad` of some declared
probe) -- reporting a never-looked-at cell, using more than Q probes, or a
malformed report floors the score to 0. For a feasible report the collected
value is  sum of true r(c) over reported cells  minus  `penalty` for every
reported cell that is in fact NOT hot (true r(c) < thr; a false positive). The
score is normalized against a deterministic uniform-grid probing baseline.

Instance regime (deterministic from the seed):
  * H, W in [40, 70] (a few thousand cells).
  * Q (probe budget) small relative to the grid: Q in [12, 24], so probes
    cannot cover the whole grid -- WHERE you probe is the whole game.
  * rad in {1, 2}: each probe sees a (2 rad + 1)^2 window.
  * a handful of Gaussian hotspots (ncomp in [3, 7]) of uneven height/spread,
    placed anywhere; background is small. This makes the hot mass concentrated
    in a few unknown regions that an adaptive, belief-driven probe schedule
    locates far better than a fixed uniform sweep.
"""
import sys
import math
import random

SCALE = 1000  # reward field is scaled to integers in [0, ~SCALE * peak]


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x42A1_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    H = rng.randint(40, 70)
    W = rng.randint(40, 70)
    Q = rng.randint(12, 24)
    rad = rng.choice([1, 1, 2])           # mostly radius-1 probes
    sigma = round(rng.uniform(0.20, 0.45), 3)   # measurement noise (rel. scale)

    # a few Gaussian hotspots of uneven height and spread.
    ncomp = rng.randint(3, 7)
    blobs = []
    for _ in range(ncomp):
        cx = rng.uniform(0.05 * H, 0.95 * H)
        cy = rng.uniform(0.05 * W, 0.95 * W)
        spread = rng.uniform(1.5, 4.5)
        height = rng.uniform(0.5, 1.0)
        blobs.append((cx, cy, spread, height))

    background = rng.uniform(0.02, 0.06)

    # build the real-valued field, then scale to integers.
    grid = [[0] * W for _ in range(H)]
    peak = 0.0
    raw = [[0.0] * W for _ in range(H)]
    for x in range(H):
        for y in range(W):
            v = background
            for (cx, cy, spread, height) in blobs:
                dx = x - cx
                dy = y - cy
                v += height * math.exp(-(dx * dx + dy * dy) / (2.0 * spread * spread))
            raw[x][y] = v
            if v > peak:
                peak = v
    if peak <= 0:
        peak = 1.0
    for x in range(H):
        for y in range(W):
            grid[x][y] = int(round(SCALE * raw[x][y] / peak))

    # hot threshold: a high fraction of the peak so only the hottest cores are
    # hot (typically a few dozen cells -- worth far more than the rest).
    thr_frac = rng.uniform(0.55, 0.72)
    thr = int(round(SCALE * thr_frac))

    # false-positive penalty per wrongly-reported cell: a sizeable fraction of a
    # hot cell's worth, so the agent must be confident before reporting.
    penalty = int(round(SCALE * rng.uniform(0.30, 0.50)))

    out = [f"{H} {W} {Q} {rad} {sigma} {thr} {penalty}"]
    for x in range(H):
        out.append(" ".join(str(grid[x][y]) for y in range(W)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
