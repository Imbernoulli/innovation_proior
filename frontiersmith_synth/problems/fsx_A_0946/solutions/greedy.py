# TIER: greedy
"""Textbook recipe: build a nearest-neighbor minimum spanning tree over the
buildings using plain Manhattan distance (geometry only -- the terrain map is
never consulted while choosing the route), connecting each new building to
the tree via a straight L-shaped segment. Depth IS computed correctly (no
wasted margin): once the route is fixed, bury every cell in it at the single
flat depth that is safe for the worst cover type the route actually crosses.
This is the "shortest path is obviously shortest" trap: it never asks
whether a longer detour through better ground would be cheaper to bury."""
import sys, json, math


def required_depth(t, kappa, depth_scale, winters):
    best = 0.0
    for w in winters:
        if t == 0:
            k = kappa["bare"]
        elif t == 1:
            k = kappa["pavement"]
        elif t == 2:
            k = kappa["snow_base"] / (1.0 + kappa["snow_sensitivity"] * w["snow"])
        else:
            k = kappa["peat"]
        d = depth_scale * math.sqrt(2.0 * k * w["fdd"])
        if d > best:
            best = d
    return best


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def main():
    inst = json.load(sys.stdin)
    grid = inst["grid"]
    buildings = [tuple(b) for b in inst["buildings"]]
    kappa = inst["kappa"]
    depth_scale = inst["depth_scale"]
    winters = inst["winters"]
    rd = {t: required_depth(t, kappa, depth_scale, winters) for t in (0, 1, 2, 3)}

    connected = [buildings[0]]
    remaining = list(range(1, len(buildings)))
    route_cells = {buildings[0]}
    while remaining:
        best = None
        for ri in remaining:
            for cb in connected:
                d = manhattan(buildings[ri], cb)
                if best is None or d < best[0]:
                    best = (d, ri, cb)
        _, ri, cb = best
        b = buildings[ri]
        r0, c0 = cb
        r1, c1 = b
        rstep = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + rstep, rstep):
            route_cells.add((r, c0))
        cstep = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + cstep, cstep):
            route_cells.add((r1, c))
        connected.append(b)
        remaining.remove(ri)

    types_used = set(grid[r][c] for (r, c) in route_cells)
    flat_depth = max(rd[t] for t in types_used)

    route = [{"r": r, "c": c, "depth": flat_depth} for (r, c) in route_cells]
    print(json.dumps({"route": route}))


if __name__ == "__main__":
    main()
