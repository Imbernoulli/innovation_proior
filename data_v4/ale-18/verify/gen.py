#!/usr/bin/env python3
"""
Instance generator for "TSP with Time Windows (soft lateness)".

Usage:  python3 gen.py SEED   ->  writes an instance to stdout.

Instance format (stdin of the solver):
    line 1:  n lambda
    line 2:  depot_x depot_y          (the depot, index 0, where the tour starts at time 0)
    next n:  x_i y_i e_i l_i          (node i, for i = 1..n; the n customers to visit)

Coordinates are integers in [0, COORD].  Travel time between two points is the
Euclidean distance (a real number).  The depot has no time window; the tour
starts at the depot at time 0, visits all n customers exactly once in some
order, and the cost is  total_travel_distance + lambda * sum(lateness).
(The tour does NOT need to return to the depot.)

Time windows are generated so that an earliest-deadline-first (EDF) ordering is
feasible-ish but far from optimal in distance: deadlines are spread across the
plausible time horizon and correlated only weakly with geography, so a good
solver must trade travel distance against lateness.
"""
import sys
import random
import math

COORD = 1000        # coordinate range [0, COORD]
N_MIN = 40
N_MAX = 60


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rng = random.Random(seed * 1_000_003 + 12345)

    n = rng.randint(N_MIN, N_MAX)
    # lambda: penalty weight per unit of lateness. Chosen so that lateness and
    # distance are comparable in magnitude (neither dominates trivially).
    lam = rng.choice([0.5, 1.0, 2.0, 4.0])

    depot = (rng.randint(0, COORD), rng.randint(0, COORD))

    pts = [(rng.randint(0, COORD), rng.randint(0, COORD)) for _ in range(n)]

    # Rough time horizon: distance the courier could plausibly cover.
    # A nearest-ish tour over n points in a COORD box has length ~ 0.7*sqrt(n)*COORD.
    horizon = 0.9 * math.sqrt(n) * COORD

    nodes = []
    for (x, y) in pts:
        # center of the time window: spread over the horizon, weakly tied to
        # the point's distance from the depot (so EDF is plausible but not free).
        d0 = math.hypot(x - depot[0], y - depot[1])
        bias = d0 / (1.4142 * COORD)              # in [0,1)
        center = (0.15 + 0.70 * rng.random() + 0.15 * bias) * horizon
        width = rng.uniform(0.05, 0.30) * horizon  # window half-width-ish
        e = max(0.0, center - width)
        l = center + width
        nodes.append((x, y, e, l))

    out = []
    out.append(f"{n} {lam:.6f}")
    out.append(f"{depot[0]} {depot[1]}")
    for (x, y, e, l) in nodes:
        out.append(f"{x} {y} {e:.6f} {l:.6f}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
