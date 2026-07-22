# TIER: trivial
"""Cheapest-K construction: ignore geometry entirely, open the K sites with the
lowest listed opening cost, then assign every household to its nearest OPEN
depot. Matches the evaluator's weak reference construction exactly."""
import sys, json, math


def main():
    inst = json.load(sys.stdin)
    sites = inst["sites"]; points = inst["points"]; k = inst["k"]
    order = sorted(range(len(sites)), key=lambda s: (sites[s][2], s))
    facilities = order[:k]

    assign = []
    for (px, py, w) in points:
        bd = None; bi = None
        for idx, s in enumerate(facilities):
            sx, sy, _ = sites[s]
            d = math.hypot(px - sx, py - sy)
            if bd is None or d < bd:
                bd = d; bi = idx
        assign.append(bi)

    print(json.dumps({"facilities": facilities, "assign": assign}))


if __name__ == "__main__":
    main()
