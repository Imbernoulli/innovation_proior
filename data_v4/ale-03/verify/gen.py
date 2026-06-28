#!/usr/bin/env python3
"""
Instance generator for "Drone Courier under Time Windows" (ale-03).

Usage:  python3 gen.py <seed>   ->  writes one instance to stdout.

Instance format (stdin of the solver):
    N T
    then N lines, request i (1-indexed):  x_i y_i r_i d_i s_i

  - N            : number of requests
  - T            : global time horizon; the drone leaves depot (id 0) at t=0
                   and must be back at the depot by time T.
  - (x_i, y_i)   : integer coordinates of request i, in [0, L]
  - [r_i, d_i]   : time window; service on i must START at time t with
                   r_i <= t <= d_i (arriving early => wait until r_i).
  - s_i          : service duration at request i.

  Depot is request id 0 located at (L/2, L/2); it has no window / no service.

Travel time between two points is the Euclidean distance rounded UP to the
next integer (ceil). Requests are placed as a mixture of spatial clusters
("delivery zones") plus uniform noise, and windows are staggered across the
horizon so that not all requests can be served -- the heuristic must choose.
"""
import sys, math, random

L = 1000          # coordinate range [0, L]

def gen(seed: int):
    rng = random.Random(seed * 1_000_003 + 12345)
    # Size class varies with the seed so the seed set spans small..large.
    N = rng.choice([200, 300, 400, 500, 600, 700, 800])
    T = rng.choice([2000, 2500, 3000, 3500, 4000])

    depot = (L // 2, L // 2)

    # --- spatial layout: a handful of Gaussian clusters + uniform sprinkle ---
    num_clusters = rng.randint(3, 7)
    clusters = []
    for _ in range(num_clusters):
        cx = rng.uniform(0.1 * L, 0.9 * L)
        cy = rng.uniform(0.1 * L, 0.9 * L)
        sd = rng.uniform(0.04 * L, 0.12 * L)
        clusters.append((cx, cy, sd))

    reqs = []
    for _ in range(N):
        if rng.random() < 0.80 and clusters:
            cx, cy, sd = rng.choice(clusters)
            x = int(round(min(L, max(0, rng.gauss(cx, sd)))))
            y = int(round(min(L, max(0, rng.gauss(cy, sd)))))
        else:
            x = rng.randint(0, L)
            y = rng.randint(0, L)

        # service duration: small fixed-ish cost
        s = rng.randint(2, 12)

        # time window: a release time scattered over the horizon and a window
        # of moderate (variable) width. Some windows are tight, some loose.
        width = rng.choice([
            rng.randint(60, 150),     # tight
            rng.randint(150, 400),    # medium
            rng.randint(400, 900),    # loose
        ])
        # release uniformly in [0, T - small], deadline = release + width (capped)
        r = rng.randint(0, max(1, T - 50))
        d = min(T - 1, r + width)
        if d <= r:
            d = min(T - 1, r + 30)
        reqs.append((x, y, r, d, s))

    out = []
    out.append(f"{N} {T}")
    for (x, y, r, d, s) in reqs:
        out.append(f"{x} {y} {r} {d} {s}")
    return "\n".join(out) + "\n"


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sys.stdout.write(gen(seed))


if __name__ == "__main__":
    main()
