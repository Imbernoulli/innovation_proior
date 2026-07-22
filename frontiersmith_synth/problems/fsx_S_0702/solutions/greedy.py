# TIER: greedy
"""Farthest-point sampling (k-center greedy): repeatedly open the candidate site
that maximizes the minimum distance to already-opened sites, then assign every
household to its nearest open depot. This is the textbook single-pass spatial
covering heuristic for facility placement. By construction it maximizes spread,
so it ALWAYS finishes opening one depot per macro district before it ever
considers a second depot inside the same district -- it locks onto the coarse
scale and cannot see that a dense population pocket deserves its own depot."""
import sys, json, math


def main():
    inst = json.load(sys.stdin)
    sites = inst["sites"]; points = inst["points"]; k = inst["k"]
    M = len(sites)

    first = min(range(M), key=lambda s: sites[s][2])
    chosen = [first]
    mind = [math.hypot(sites[s][0] - sites[first][0], sites[s][1] - sites[first][1]) for s in range(M)]
    chosen_set = {first}
    while len(chosen) < k:
        best = None; bestd = -1.0
        for s in range(M):
            if s in chosen_set:
                continue
            if mind[s] > bestd or (mind[s] == bestd and (best is None or s < best)):
                bestd = mind[s]; best = s
        chosen.append(best); chosen_set.add(best)
        for s in range(M):
            d = math.hypot(sites[s][0] - sites[best][0], sites[s][1] - sites[best][1])
            if d < mind[s]:
                mind[s] = d

    assign = []
    for (px, py, w) in points:
        bd = None; bi = None
        for idx, s in enumerate(chosen):
            sx, sy, _ = sites[s]
            d = math.hypot(px - sx, py - sy)
            if bd is None or d < bd:
                bd = d; bi = idx
        assign.append(bi)

    print(json.dumps({"facilities": chosen, "assign": assign}))


if __name__ == "__main__":
    main()
