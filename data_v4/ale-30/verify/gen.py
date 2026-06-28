#!/usr/bin/env python3
"""
Instance generator for ale-30: Tower Placement for Signal Coverage.

Usage: python3 gen.py <seed>   ->  writes one instance to stdout.

An instance:
  - n demand nodes, m candidate tower sites, placed on a [0, L) x [0, L) plane.
  - Each demand i has a required signal req[i] (a positive real, printed with 6 decimals).
  - A tower at site j delivers power p[j][i] to demand i, distance-decayed:
        p[j][i] = P0 / (1 + (dist(j,i) / R)^2)        (Cauchy / Lorentzian decay)
    truncated to 0 when dist(j,i) > CUT (so far towers contribute nothing).
  - We GUARANTEE feasibility: selecting ALL m sites satisfies every demand
    (req[i] is set to a fraction of the total power demand i would receive from
     all sites), so the trivial "select every site" output is always feasible
     and the feasibility->0 floor is meaningful.

Output format (stdin of the solver), all on whitespace-separated tokens:
    n m
    req[0] req[1] ... req[n-1]
    then n lines, line i has m values: p[0][i] p[1][i] ... p[m-1][i]
        (i.e. the power site j delivers to demand i; row = demand, col = site)
"""
import sys
import math
import random


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(seed * 2654435761 + 12345)

    L = 1000.0
    # Problem size: vary a little with the seed so the seed set spans regimes,
    # but keep it bounded so the solver stays fast.
    n = rng.randint(180, 260)   # demand nodes
    m = rng.randint(90, 140)    # candidate sites
    P0 = 100.0                  # peak power at zero distance
    R = rng.uniform(120.0, 200.0)   # decay radius
    CUT = 3.5 * R               # hard coverage cutoff

    # Demands clustered into a few hotspots; sites scattered (some near, some far).
    n_clusters = rng.randint(3, 6)
    centers = [(rng.uniform(0, L), rng.uniform(0, L)) for _ in range(n_clusters)]

    dem = []
    for _ in range(n):
        cx, cy = rng.choice(centers)
        x = min(L - 1e-6, max(0.0, rng.gauss(cx, L * 0.10)))
        y = min(L - 1e-6, max(0.0, rng.gauss(cy, L * 0.10)))
        dem.append((x, y))

    sites = []
    for _ in range(m):
        if rng.random() < 0.6:
            cx, cy = rng.choice(centers)
            x = min(L - 1e-6, max(0.0, rng.gauss(cx, L * 0.18)))
            y = min(L - 1e-6, max(0.0, rng.gauss(cy, L * 0.18)))
        else:
            x = rng.uniform(0, L)
            y = rng.uniform(0, L)
        sites.append((x, y))

    # Power matrix p[i][j] = power site j gives demand i.
    P = [[0.0] * m for _ in range(n)]
    total_per_demand = [0.0] * n
    for i in range(n):
        dx, dy = dem[i]
        for j in range(m):
            sx, sy = sites[j]
            d = math.hypot(dx - sx, dy - sy)
            if d > CUT:
                val = 0.0
            else:
                val = P0 / (1.0 + (d / R) ** 2)
            P[i][j] = val
            total_per_demand[i] += val

    # Requirement = fraction of the all-sites-on total power, so all-on is feasible.
    # Fraction varies per demand to create heterogeneous tightness.
    req = []
    for i in range(n):
        frac = rng.uniform(0.30, 0.62)
        r = total_per_demand[i] * frac
        # Numerical guard: keep it strictly below the all-on total.
        r = min(r, total_per_demand[i] * 0.95)
        req.append(r)

    out = []
    out.append(f"{n} {m}")
    out.append(" ".join(f"{r:.6f}" for r in req))
    for i in range(n):
        out.append(" ".join(f"{P[i][j]:.6f}" for j in range(m)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
