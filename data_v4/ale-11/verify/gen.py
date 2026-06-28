#!/usr/bin/env python3
"""
Instance generator for "Vehicle Dispatch over Time" (ale-11).

Usage:  python3 gen.py <seed>   ->  writes one instance to stdout.

Instance format (stdin of the solver):
    V N T
    then V lines: sx sy            -- starting cell of vehicle v (0-indexed v)
    then N lines: px py qx qy r e  -- request i (0-indexed i):
                                       pickup (px,py), dropoff (qx,qy),
                                       release time r, expiry time e.

  - V            : number of vehicles.
  - N            : number of ride requests.
  - T            : global time horizon (integer ticks). Everything happens in
                   [0, T]; any pickup must START at a time t with r <= t <= e.
  - grid is L x L (cells 0..L-1 on each axis); L is fixed below.
  - travel time between two cells = Manhattan distance |dx|+|dy| (integer ticks).
    A vehicle that finishes a ride at cell c at time t is free at time t and at
    cell c; to serve a request it must first drive (empty) to the pickup, then
    drive (loaded) to the dropoff. The pickup is the START of service.

  A request i is *fulfillable* by a free vehicle at (vx,vy,vt) iff
        start = max(vt + manhattan((vx,vy),(px,py)), r) <= e
  and after pickup the vehicle drives to the dropoff, finishing at
        finish = start + manhattan((px,py),(qx,qy)),
  which must satisfy finish <= T. The vehicle is then free again at the dropoff.

Requests are released over the whole horizon (so dispatch is a moving target),
laid out as a mixture of spatial demand hotspots plus uniform sprinkle, with a
mix of tight and loose expiry windows so that not every request can be served:
the dispatcher must CHOOSE which rides to take and which vehicle takes each.
"""
import sys, random

L = 200                # grid is L x L


def gen(seed: int):
    rng = random.Random(seed * 1_000_003 + 777)

    # Size class varies with the seed so the seed set spans small..large.
    V = rng.choice([5, 8, 10, 12, 15, 20])
    N = rng.choice([150, 200, 300, 400, 500, 600])
    T = rng.choice([400, 500, 600, 800, 1000])

    # --- spatial demand hotspots + uniform sprinkle ----------------------
    num_hot = rng.randint(3, 7)
    hots = []
    for _ in range(num_hot):
        cx = rng.uniform(0.1 * L, 0.9 * L)
        cy = rng.uniform(0.1 * L, 0.9 * L)
        sd = rng.uniform(0.04 * L, 0.14 * L)
        hots.append((cx, cy, sd))

    def sample_cell():
        if rng.random() < 0.75 and hots:
            cx, cy, sd = rng.choice(hots)
            x = int(round(min(L - 1, max(0, rng.gauss(cx, sd)))))
            y = int(round(min(L - 1, max(0, rng.gauss(cy, sd)))))
        else:
            x = rng.randint(0, L - 1)
            y = rng.randint(0, L - 1)
        return x, y

    # vehicle starts
    vstarts = [sample_cell() for _ in range(V)]

    reqs = []
    for _ in range(N):
        px, py = sample_cell()
        qx, qy = sample_cell()
        # release uniformly over the horizon (leave room to actually serve)
        r = rng.randint(0, max(1, T - 40))
        # window width: a mix of tight / medium / loose
        width = rng.choice([
            rng.randint(20, 60),     # tight
            rng.randint(60, 160),    # medium
            rng.randint(160, 360),   # loose
        ])
        e = min(T, r + width)
        if e <= r:
            e = min(T, r + 20)
        reqs.append((px, py, qx, qy, r, e))

    out = [f"{V} {N} {T}"]
    for (sx, sy) in vstarts:
        out.append(f"{sx} {sy}")
    for (px, py, qx, qy, r, e) in reqs:
        out.append(f"{px} {py} {qx} {qy} {r} {e}")
    return "\n".join(out) + "\n"


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sys.stdout.write(gen(seed))


if __name__ == "__main__":
    main()
