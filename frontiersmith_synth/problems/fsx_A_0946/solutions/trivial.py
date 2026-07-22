# TIER: trivial
"""Laziest safe recipe: connect the buildings in the order they were given
by straight L-shaped (row-then-column) segments, and bury the WHOLE thing at
one flat depth big enough to be safe no matter what cover type it crosses:
1.5x the single worst required cover-type depth on the map. Never computes
a required depth per cell, never considers rerouting through better ground."""
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


def main():
    inst = json.load(sys.stdin)
    buildings = [tuple(b) for b in inst["buildings"]]
    kappa = inst["kappa"]
    depth_scale = inst["depth_scale"]
    winters = inst["winters"]

    rd = {t: required_depth(t, kappa, depth_scale, winters) for t in (0, 1, 2, 3)}
    flat_depth = 1.5 * max(rd.values())

    route_cells = {buildings[0]}
    for i in range(len(buildings) - 1):
        r0, c0 = buildings[i]
        r1, c1 = buildings[i + 1]
        rstep = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + rstep, rstep):
            route_cells.add((r, c0))
        cstep = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + cstep, cstep):
            route_cells.add((r1, c))

    route = [{"r": r, "c": c, "depth": flat_depth} for (r, c) in route_cells]
    print(json.dumps({"route": route}))


if __name__ == "__main__":
    main()
