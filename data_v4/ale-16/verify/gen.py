#!/usr/bin/env python3
"""Instance generator for "Prize-Collecting Patrol" (ALE-Bench heuristic optimization).

Usage:
    python3 gen.py <seed>

Writes one instance to stdout in the format:

    n
    dx dy
    x_0 y_0 prize_0
    x_1 y_1 prize_1
    ...
    x_{n-1} y_{n-1} prize_{n-1}

where:
  * n is the number of optional prize nodes (chosen deterministically from the
    seed in [400, 1200]),
  * (dx, dy) are the integer coordinates of the depot (always visited, no prize),
  * each node i has integer coordinates 0 <= x_i, y_i <= 1_000_000 and an integer
    prize_i > 0.

Design intent: a Prize-Collecting TSP instance where a *strict subset* of nodes is
worth visiting. Coordinates mix a few 2-D Gaussian clusters (rich neighbourhoods)
with a uniform background of far-flung, low-prize nodes. Prizes are correlated
neither perfectly with nor independently of geography: cluster nodes tend to carry
larger prizes, background nodes carry small ones, and there is per-node noise. This
makes the "skip the loser" decision (include a node only if its prize exceeds the
detour cost of inserting it) genuinely load-bearing -- visiting everything is a
losing strategy, and so is visiting almost nothing.
"""
import sys
import random

SIDE = 1_000_000  # coordinate grid is [0, SIDE] in each axis


def main() -> None:
    if len(sys.argv) < 2:
        sys.stderr.write("usage: gen.py <seed>\n")
        sys.exit(1)
    seed = int(sys.argv[1])
    rng = random.Random(0x9C0FFEE ^ (seed * 2654435761 & 0xFFFFFFFF))

    # number of optional nodes: deterministic from the seed, in [400, 1200]
    n = rng.randint(400, 1200)

    # depot somewhere near the middle-ish (still random)
    dx = rng.randint(0, SIDE)
    dy = rng.randint(0, SIDE)

    # cluster structure
    num_clusters = rng.randint(3, 8)
    centers = [(rng.uniform(0, SIDE), rng.uniform(0, SIDE)) for _ in range(num_clusters)]
    spreads = [rng.uniform(0.02, 0.08) * SIDE for _ in range(num_clusters)]
    bg_frac = rng.uniform(0.35, 0.65)  # fraction drawn uniformly (background)

    # Prize scale -- the load-bearing design choice. Profit = (sum prizes) -
    # (Euclidean travel), both in coordinate units. The marginal cost of inserting
    # a node into an existing tour is roughly TWICE its distance to the nearest
    # tour node; for "skip the loser" to be a real decision, many nodes must have a
    # prize BELOW that marginal cost. With n nodes in a 1e6-side box, the typical
    # nearest-neighbour spacing is ~ SIDE/sqrt(n) (tens of thousands of units), and
    # ISOLATED background nodes sit much farther from any tour. So we keep prizes
    # MODEST: cluster nodes carry a prize comparable to a few NN-spacings (worth the
    # short detour inside a dense cluster), while sparse background nodes carry a
    # prize that usually does NOT cover their long detour -- they are the losers a
    # good solver must skip. `unit` ties the prize scale to the instance geometry.
    unit = SIDE / max(1.0, (n ** 0.5))  # ~ typical nearest-neighbour spacing
    pts = []
    seen = set()
    attempts = 0
    while len(pts) < n and attempts < 60 * n:
        attempts += 1
        if rng.random() < bg_frac:
            x = rng.uniform(0, SIDE)
            y = rng.uniform(0, SIDE)
            # background nodes: small prize (often below their detour cost)
            base = rng.uniform(0.3, 2.0) * unit
        else:
            c = rng.randrange(num_clusters)
            cx, cy = centers[c]
            s = spreads[c]
            x = rng.gauss(cx, s)
            y = rng.gauss(cy, s)
            # cluster nodes: larger prize, usually worth visiting when nearby
            base = rng.uniform(1.5, 6.0) * unit
        xi = int(round(x))
        yi = int(round(y))
        if xi < 0:
            xi = 0
        elif xi > SIDE:
            xi = SIDE
        if yi < 0:
            yi = 0
        elif yi > SIDE:
            yi = SIDE
        key = (xi, yi)
        if key in seen or (xi == dx and yi == dy):
            continue  # distinct coords (and distinct from depot)
        seen.add(key)
        prize = int(round(base))
        if prize < 1:
            prize = 1
        pts.append((xi, yi, prize))

    # top up uniformly if clustering collided too much
    while len(pts) < n:
        xi = rng.randint(0, SIDE)
        yi = rng.randint(0, SIDE)
        key = (xi, yi)
        if key in seen or (xi == dx and yi == dy):
            continue
        seen.add(key)
        prize = int(round(rng.uniform(0.3, 2.0) * unit))
        if prize < 1:
            prize = 1
        pts.append((xi, yi, prize))

    out = [str(n), f"{dx} {dy}"]
    out.extend(f"{x} {y} {p}" for (x, y, p) in pts)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
