# TIER: strong
"""Adaptive destroy/reinsert local search that sizes every move to LOCAL demand
density, rather than a fixed neighborhood.

1. demand-cluster-diagnosis: for every household, measure the distance to its
   5th-nearest neighboring household. Small -> it sits inside a tight, dense
   pocket. Large -> it is in the thin countryside.
2. Start from a farthest-point-sampling layout (same as `greedy`), then repeat:
     - find the currently worst-served household ("hot point": largest
       weight * distance-to-depot regret), with occasional random exploration.
     - density-scaled destroy radius: radius = clamp(alpha * local_scale, rmin,
       rmax). In a dense pocket this is SMALL -> a single surgical removal near
       the hot point (or, if none is that close yet, the nearest open depot is
       pulled in to be relocated). In sparse countryside this is LARGE -> a
       wider sweep that can free up several depots at once.
     - reinsert the freed depot(s) at the best nearby not-yet-open site(s)
       (greedy marginal-cost repair, restricted to a small local pool).
3. threshold-acceptance schedule: accept the destroy/reinsert move whenever it
   does not worsen cost by more than a threshold that shrinks linearly from an
   initial slack (3% of the current cost) down to 0 over the iteration budget --
   early on this lets the search hop OUT of the coarse-scale local optimum that
   pure greedy spread never escapes; late on it behaves like strict descent.

This is fundamentally different from "greedy + more iterations": greedy never
revisits a placement once made, so it can never discover that splitting a
dense pocket off from its district depot pays for itself.
"""
import sys, json, math, random


def nearest_cost(sites, points, facs):
    total = sum(sites[s][2] for s in facs)
    assign = []
    for (px, py, w) in points:
        bd = None; bi = None
        for idx, s in enumerate(facs):
            sx, sy, _ = sites[s]
            d = math.hypot(px - sx, py - sy)
            if bd is None or d < bd:
                bd = d; bi = idx
        total += w * bd
        assign.append(bi)
    return total, assign


def farthest_point_sampling(sites, k):
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
    return chosen


def knn_scale(points, kk=5):
    N = len(points)
    out = []
    for i in range(N):
        px, py, _ = points[i]
        ds = []
        for j in range(N):
            if j == i:
                continue
            qx, qy, _ = points[j]
            ds.append(math.hypot(px - qx, py - qy))
        ds.sort()
        idx = min(kk, len(ds) - 1) if ds else 0
        out.append(ds[idx] if ds else 50.0)
    return out


def solve(sites, points, k, iters=90, alpha=2.5, rmin=15.0, rmax=420.0, pool_cap=6):
    M = len(sites)
    scale = knn_scale(points, kk=5)
    rng = random.Random(20260702)   # fixed seed: fully deterministic, instance-independent

    cur = farthest_point_sampling(sites, k)
    cur_cost, cur_assign = nearest_cost(sites, points, cur)
    best_cost, best_facs = cur_cost, list(cur)
    T0 = 0.03 * cur_cost

    for t in range(iters):
        thr = T0 * max(0.0, 1.0 - t / max(1, iters))
        dists = [math.hypot(points[i][0] - sites[cur[cur_assign[i]]][0],
                             points[i][1] - sites[cur[cur_assign[i]]][1])
                  for i in range(len(points))]
        regret = [points[i][2] * dists[i] for i in range(len(points))]
        if rng.random() < 0.15:
            hot = rng.randrange(len(points))
        else:
            hot = max(range(len(points)), key=lambda i: regret[i])
        hx, hy, _ = points[hot]

        radius = min(max(alpha * scale[hot], rmin), rmax)   # density-scaled destroy radius
        open_set = set(cur)
        to_remove = [p for p, s in enumerate(cur) if math.hypot(sites[s][0] - hx, sites[s][1] - hy) <= radius]
        if not to_remove:
            to_remove = [min(range(len(cur)), key=lambda p: math.hypot(sites[cur[p]][0] - hx, sites[cur[p]][1] - hy))]
        remaining = [s for p, s in enumerate(cur) if p not in to_remove]

        not_open = [s for s in range(M) if s not in open_set]
        pool = sorted(not_open, key=lambda s: math.hypot(sites[s][0] - hx, sites[s][1] - hy))[:pool_cap]
        if len(pool) < len(to_remove):
            pool = sorted(not_open, key=lambda s: math.hypot(sites[s][0] - hx, sites[s][1] - hy))

        chosen = []
        for _slot in range(len(to_remove)):
            avail = [s for s in pool if s not in chosen] or [s for s in not_open if s not in chosen]
            bestc, bests = None, None
            for cand in avail:
                trial = remaining + chosen + [cand]
                c, _ = nearest_cost(sites, points, trial)
                if bestc is None or c < bestc:
                    bestc, bests = c, cand
            chosen.append(bests)

        new_facs = remaining + chosen
        new_cost, new_assign = nearest_cost(sites, points, new_facs)

        if new_cost <= cur_cost + thr:      # threshold-acceptance schedule
            cur, cur_cost, cur_assign = new_facs, new_cost, new_assign
            if cur_cost < best_cost:
                best_cost, best_facs = cur_cost, list(cur)

    _, best_assign = nearest_cost(sites, points, best_facs)
    return best_facs, best_assign


def main():
    inst = json.load(sys.stdin)
    sites = inst["sites"]; points = inst["points"]; k = inst["k"]
    facs, assign = solve(sites, points, k)
    print(json.dumps({"facilities": facs, "assign": assign}))


if __name__ == "__main__":
    main()
