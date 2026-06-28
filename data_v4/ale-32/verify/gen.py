#!/usr/bin/env python3
"""Instance generator for "Capacitated k-Means" (ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the format:

    n k cap
    x_0 y_0
    x_1 y_1
    ...
    x_{n-1} y_{n-1}

Meaning: n points in the integer plane must be partitioned into k clusters. Each
cluster may hold AT MOST `cap` points (a hard per-cluster cardinality cap). A
cluster's representative is the centroid (mean) of its assigned points, computed
in real arithmetic. The objective is to MINIMIZE the total squared Euclidean
distance from every point to the centroid of the cluster it is assigned to (see
score.py / context.md for the exact rule and the feasibility -> 0 floor).

Instance regime (deterministic from the seed):
  * n points in [400, 900].
  * k clusters in [6, 14].
  * cap is set so that k * cap is only a little larger than n -- the caps are
    BINDING (k * cap in [1.05 * n, 1.30 * n]). This is exactly the regime where
    the cap turns an ordinary Lloyd / k-means iteration into a TRANSPORTATION
    problem: a few clusters would love to grab more than `cap` points and must be
    forced to overflow into a neighbour, so a naive nearest-centroid assignment is
    infeasible and a naive greedy cap-repair leaves a lot of cost on the table.
  * The points are drawn from a mixture of `ncomp` 2D Gaussian blobs (ncomp is
    near k but deliberately not equal -- sometimes a few clusters share a blob,
    sometimes a blob is split), with per-blob spread and weight, clipped to a
    [0, COORD] integer box. The uneven blob populations are what make the caps
    bite unevenly across the plane.
"""
import sys
import random

COORD = 10000  # coordinate box is [0, COORD] in both axes


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x32A1_0000 ^ (seed * 2654435761 & 0xFFFFFFFF))

    n = rng.randint(400, 900)
    k = rng.randint(6, 14)

    # caps are binding: total capacity only modestly exceeds n.
    slack = rng.uniform(1.05, 1.30)
    cap = max(1, int((n * slack) / k + 0.999))
    # guarantee feasibility of the cap: k * cap must be >= n.
    while k * cap < n:
        cap += 1

    # mixture of Gaussian blobs; ncomp near k but not equal.
    ncomp = max(2, k + rng.randint(-3, 3))
    blobs = []
    for _ in range(ncomp):
        cx = rng.uniform(0.08 * COORD, 0.92 * COORD)
        cy = rng.uniform(0.08 * COORD, 0.92 * COORD)
        spread = rng.uniform(0.03 * COORD, 0.12 * COORD)
        weight = rng.uniform(0.4, 1.6)
        blobs.append((cx, cy, spread, weight))
    wtot = sum(b[3] for b in blobs)

    pts = []
    for _ in range(n):
        # pick a blob by weight
        r = rng.uniform(0, wtot)
        acc = 0.0
        bi = 0
        for j, b in enumerate(blobs):
            acc += b[3]
            if r <= acc:
                bi = j
                break
        cx, cy, spread, _ = blobs[bi]
        x = int(round(rng.gauss(cx, spread)))
        y = int(round(rng.gauss(cy, spread)))
        x = min(COORD, max(0, x))
        y = min(COORD, max(0, y))
        pts.append((x, y))

    out = [f"{n} {k} {cap}"]
    out.extend(f"{x} {y}" for (x, y) in pts)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
