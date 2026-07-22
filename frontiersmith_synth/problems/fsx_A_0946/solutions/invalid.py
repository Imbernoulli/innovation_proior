# TIER: invalid
"""Deliberately broken: connects the buildings with straight L-paths (looks
plausible) but buries everything at a fixed shallow depth that ignores the
frost physics entirely, so it freezes on essentially every instance."""
import sys, json


def main():
    inst = json.load(sys.stdin)
    buildings = [tuple(b) for b in inst["buildings"]]

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

    route = [{"r": r, "c": c, "depth": 0.05} for (r, c) in route_cells]
    print(json.dumps({"route": route}))


if __name__ == "__main__":
    main()
