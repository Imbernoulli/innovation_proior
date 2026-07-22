#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE TRAIN sample to stdout.

Reverse-engineering why a drone fleet slows near itself.

Hidden law (lives in gen.py AND verify.py identically; NEVER printed):
    realized = commanded * outer( S )         S = sum over sensed neighbors of kernel(dist)
    kernel(d) = a / (d*d + c)                 -- pairwise interference kernel (a,c hidden, seeded per testId)
    outer(S)  = 1 / (1 + S)                   -- saturating aggregate slowdown (bounded in (0,1])

Each row is one drone's local sensor reading at one instant: its commanded
speed, the distances to every drone within its sensing radius, and its
REALIZED speed (kernel+outer law plus small measurement noise).

The TRAIN rows are logged from THREE kinds of flights, all sharing the same
schema (the row's neighbor COUNT tells you which kind it is -- no extra tag
is printed):
  - ISOLATED-PAIR passes (n=1 neighbor): two drones alone at a swept range of
    separations. A clean 1-D slice: only ONE kernel term contributes, so S is
    small and the pairwise falloff is visible with the least confounding.
  - CONTROLLED CLUSTER holds (n=2..7, all listed distances nearly equal): a
    small ring formation held at a fixed spacing while the ring SIZE is
    varied. Same per-neighbor kernel value, known multiplicity -- this is
    what exposes whether the aggregate slowdown keeps growing linearly with
    neighbor count or SATURATES.
  - ORGANIC mixed-fleet flights (n=1..7, irregular distances): everyday
    logged flights, noisier and structurally uninformative on their own.

Only fleets of 3-8 drones are ever logged for TRAIN. Held-out grading (inside
verify.py only) uses 20-40 drone swarms flown roughly 3x DENSER -- a genuine
density extrapolation, never printed here.
"""
import sys, random, math


def law_params(t):
    """Hidden kernel constants for this test id (identical copy lives in verify.py)."""
    rng = random.Random(9130007 + t * 7919)
    a = rng.uniform(0.85, 1.55)
    c = rng.uniform(0.06, 0.24)
    return a, c


def kernel(d, a, c):
    return a / (d * d + c)


def outer(S):
    return 1.0 / (1.0 + S)


def make_row(rng, v, dists, a, c, sigma):
    S = sum(kernel(d, a, c) for d in dists)
    y = v * outer(S) + rng.gauss(0.0, sigma)
    y = max(0.0, y)
    return len(dists), v, y, dists


def gen_pair_rows(rng, n_pairs, a, c, sigma, dmin, dmax):
    rows = []
    for _ in range(n_pairs):
        v = rng.uniform(2.0, 6.0)
        d = rng.uniform(dmin, dmax)
        rows.append(make_row(rng, v, [d], a, c, sigma))
    return rows


def gen_cluster_rows(rng, n_clusters, a, c, sigma, dmin, dmax):
    rows = []
    for _ in range(n_clusters):
        k = rng.randint(2, 7)          # ring size -> k neighbors, one target agent
        d0 = rng.uniform(dmin, dmax)
        v = rng.uniform(2.0, 6.0)
        # near-equal spacing (tiny jitter so it isn't perfectly degenerate)
        dists = [d0 * (1.0 + rng.uniform(-0.02, 0.02)) for _ in range(k)]
        rows.append(make_row(rng, v, dists, a, c, sigma))
    return rows


def gen_organic_rows(rng, n_rows, a, c, sigma, dmin, dmax):
    rows = []
    for _ in range(n_rows):
        k = rng.randint(1, 7)
        v = rng.uniform(2.0, 6.0)
        dists = [rng.uniform(dmin, dmax) for _ in range(k)]
        rows.append(make_row(rng, v, dists, a, c, sigma))
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rng = random.Random(555001 + t * 104729)
    a, c = law_params(t)

    # difficulty ladder: later testIds -> a bit more noise, a bit less data
    sigma = 0.028 + 0.006 * (t - 1)
    n_pairs = max(18, 34 - 2 * (t - 1))
    n_clusters = max(14, 26 - 2 * (t - 1))
    n_organic = max(30, 60 - 3 * (t - 1))

    dmin, dmax = 0.45, 3.2

    rows = []
    rows += gen_pair_rows(rng, n_pairs, a, c, sigma, dmin, dmax)
    rows += gen_cluster_rows(rng, n_clusters, a, c, sigma, dmin, dmax)
    rows += gen_organic_rows(rng, n_organic, a, c, sigma, dmin, dmax)
    rng.shuffle(rows)

    out = ["%d %d" % (len(rows), t)]
    for n, v, y, dists in rows:
        parts = ["%d" % n, "%.6f" % v, "%.6f" % y] + ["%.6f" % d for d in dists]
        out.append(" ".join(parts))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
