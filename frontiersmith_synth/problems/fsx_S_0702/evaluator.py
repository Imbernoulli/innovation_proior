#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0702 -- "Meridian County Micro-Depot Network"
(family: adaptive-lns-facility-placement; format B, quality-metric).

THEME.  Meridian County must open a batch of last-mile parcel MICRO-DEPOTS to serve
N households/blocks, each with a daily parcel WEIGHT (demand volume) and a fixed
(x, y) location.  Depots may only be built at one of M CANDIDATE SITES, each with its
own construction/lease cost.  You must pick exactly K sites to open and assign every
household to one open depot; the county pays the sum of the K opening costs plus, for
every household, weight * (Euclidean distance to its assigned depot) -- the daily
delivery-run cost.  Minimize total cost.

The county's settlement pattern is HIDDEN MULTI-SCALE CLUSTERING: a handful of
regions (macro districts), each containing one or more TIGHT population pockets
(dense apartment blocks / hamlets) at a finer scale, plus households scattered
thinly across the open countryside (noise).  Candidate sites exist near every
pocket, near every district center, and scattered across the countryside.  Because
K is smaller than "one depot per pocket", the county must decide which pockets
deserve their OWN depot and which can share a coarser, district-level depot --
a genuine multi-scale trade-off with no single clean optimum.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n_demand": N, "n_sites": M, "k": K,
             "points": [[x, y, w], ...],   # length N; household location + weight
             "sites":  [[x, y, cost], ...]} # length M; candidate site + open cost
  stdout: ONE JSON object:
            {"facilities": [s_0, ..., s_{K-1}],   # K DISTINCT site indices in [0, M)
             "assign":     [f_0, ..., f_{N-1}]}   # f_i in [0, K): position in `facilities`
                                                   # that household i is served by

  VALID iff `facilities` has exactly K distinct integers in [0, M) and `assign` has
  exactly N integers in [0, K). Wrong length/type, duplicates, out-of-range, a crash,
  a timeout, or non-JSON output -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  cost(plan) = sum of opened sites' costs +
sum_i weight_i * dist(household_i, its assigned depot).  Per instance the evaluator
computes two references (using the FULL, honest problem -- no cheating, just more
search budget than a solver can spend inside its own time limit):
    q_base = cost of opening the K CHEAPEST sites (cost-only, geometry-blind), with
             every household assigned to its nearest OPEN depot (best possible
             assignment for that fixed depot set)  -- a weak, easy-to-match reference.
    q_ref  = cost of an internal high-effort adaptive destroy/reinsert search (more
             restarts + iterations than the time budget allows a solver), again with
             nearest-depot assignment -- a strong, generally-unreachable reference.
    q_cand = cost of the candidate's own submitted (facilities, assign).
  normalized with an affine anchor (weak baseline -> 0.1, strong internal ref -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_base - q_cand) / max(1e-9, q_base - q_ref), 0, 1 )

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance. The references and
scoring live entirely in this parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math, random
import isorun

DOMAIN = 1000.0


# =============================== instance family ============================
def _build_instance(seed, n_macro, sub_lo, sub_hi, pts_lo, pts_hi,
                     meso_lo, meso_hi, micro_lo, micro_hi,
                     noise_n, extra_sites_n, k, cost_lo, cost_hi):
    """Deterministic multi-scale demand family.

    n_macro coarse DISTRICTS are scattered across the county; each district holds
    one or more tight POCKETS (dense sub-clusters) at a finer scale; a handful of
    households are scattered thinly as countryside noise. Candidate sites exist at
    every pocket (plus a nearby decoy), at every district center, and scattered
    across the countryside -- so the search must choose which scale to serve at.
    """
    rng = random.Random(seed)
    macro_centers = []
    tries = 0
    while len(macro_centers) < n_macro and tries < 4000:
        tries += 1
        c = (rng.uniform(80, DOMAIN - 80), rng.uniform(80, DOMAIN - 80))
        if all(math.hypot(c[0] - m[0], c[1] - m[1]) > 260 for m in macro_centers):
            macro_centers.append(c)
    while len(macro_centers) < n_macro:
        macro_centers.append((rng.uniform(80, DOMAIN - 80), rng.uniform(80, DOMAIN - 80)))

    points = []          # (x, y, w)
    sub_true_sites = []  # (x, y) exact pocket centers
    sites = []           # [x, y, cost]

    for (mx, my) in macro_centers:
        n_sub = rng.randint(sub_lo, sub_hi)
        for _ in range(n_sub):
            ang = rng.uniform(0.0, 2.0 * math.pi)
            meso = rng.uniform(meso_lo, meso_hi)
            scx = min(max(mx + meso * math.cos(ang), 20.0), DOMAIN - 20.0)
            scy = min(max(my + meso * math.sin(ang), 20.0), DOMAIN - 20.0)
            sub_true_sites.append((scx, scy))
            micro = rng.uniform(micro_lo, micro_hi)
            npts = rng.randint(pts_lo, pts_hi)
            for _ in range(npts):
                px = min(max(scx + rng.uniform(-micro, micro), 0.0), DOMAIN)
                py = min(max(scy + rng.uniform(-micro, micro), 0.0), DOMAIN)
                w = rng.randint(1, 3)
                points.append((px, py, w))
        # candidate site at the district center itself (coarse option)
        cost = rng.uniform(cost_lo * 0.9, cost_hi * 0.9)
        sites.append([round(mx, 2), round(my, 2), round(cost, 1)])

    for (scx, scy) in sub_true_sites:
        cost = rng.uniform(cost_lo, cost_hi)
        sites.append([round(scx, 2), round(scy, 2), round(cost, 1)])
        dx = min(max(scx + rng.uniform(-40, 40), 10.0), DOMAIN - 10.0)
        dy = min(max(scy + rng.uniform(-40, 40), 10.0), DOMAIN - 10.0)
        cost2 = rng.uniform(cost_lo, cost_hi)
        sites.append([round(dx, 2), round(dy, 2), round(cost2, 1)])

    for _ in range(noise_n):
        points.append((rng.uniform(10, DOMAIN - 10), rng.uniform(10, DOMAIN - 10), 1))

    for _ in range(extra_sites_n):
        ex, ey = rng.uniform(10, DOMAIN - 10), rng.uniform(10, DOMAIN - 10)
        cost = rng.uniform(cost_lo, cost_hi)
        sites.append([round(ex, 2), round(ey, 2), round(cost, 1)])

    rng.shuffle(points)
    rng.shuffle(sites)

    return {
        "name": f"depot{seed}",
        "n_demand": len(points),
        "n_sites": len(sites),
        "k": k,
        "points": [[round(p[0], 2), round(p[1], 2), p[2]] for p in points],
        "sites": sites,
    }


def _build_instances():
    specs = [
        # seed, n_macro, sub_lo, sub_hi, pts_lo, pts_hi, meso_lo, meso_hi,
        # micro_lo, micro_hi, noise_n, extra_sites_n, k, cost_lo, cost_hi
        (702001, 3, 2, 2, 10, 16, 90, 150, 10, 18, 8, 8, 5, 150, 320),
        (702002, 4, 1, 3, 8, 14, 80, 140, 9, 16, 10, 10, 6, 150, 300),
        (702003, 3, 2, 3, 14, 20, 100, 160, 8, 14, 12, 12, 6, 180, 350),
        (702004, 4, 1, 2, 8, 12, 70, 130, 10, 20, 6, 6, 6, 140, 280),
        (702005, 3, 2, 2, 16, 22, 110, 170, 7, 12, 14, 14, 5, 200, 380),
        (702006, 5, 1, 2, 6, 10, 70, 120, 10, 18, 10, 8, 7, 150, 300),
        (702007, 4, 1, 3, 12, 18, 90, 150, 8, 14, 14, 14, 6, 170, 320),
        (702008, 3, 2, 2, 9, 14, 85, 140, 11, 19, 8, 8, 5, 160, 300),
        # held-out, larger / harder
        (702011, 5, 1, 3, 10, 16, 90, 160, 9, 16, 16, 16, 8, 160, 320),
        (702012, 4, 2, 3, 14, 20, 100, 170, 7, 13, 18, 18, 7, 190, 350),
    ]
    return [_build_instance(*s) for s in specs]


# =============================== geometry / cost =============================
def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _nearest_cost(inst, facs):
    """Given an OPEN facility set (list of site indices), return (cost, assign)
    under the (always cost-optimal, for a fixed facility set) nearest-depot
    assignment."""
    sites = inst["sites"]; points = inst["points"]
    total = sum(sites[s][2] for s in facs)
    assign = []
    for (px, py, w) in points:
        bd = None; bi = None
        for idx, s in enumerate(facs):
            sx, sy, _ = sites[s]
            dd = math.hypot(px - sx, py - sy)
            if bd is None or dd < bd:
                bd = dd; bi = idx
        total += w * bd
        assign.append(bi)
    return total, assign


def _score_answer(inst, ans):
    """Strictly validate + cost the CANDIDATE's own submitted answer. None -> invalid."""
    if not isinstance(ans, dict):
        return None
    facs = ans.get("facilities"); assign = ans.get("assign")
    if not isinstance(facs, list) or not isinstance(assign, list):
        return None
    k = inst["k"]; M = inst["n_sites"]; N = inst["n_demand"]
    if len(facs) != k:
        return None
    seen = set()
    for s in facs:
        if isinstance(s, bool) or not isinstance(s, int):
            return None
        if s < 0 or s >= M or s in seen:
            return None
        seen.add(s)
    if len(assign) != N:
        return None
    sites = inst["sites"]; points = inst["points"]
    total = sum(sites[s][2] for s in facs)
    for i, a in enumerate(assign):
        if isinstance(a, bool) or not isinstance(a, int):
            return None
        if a < 0 or a >= k:
            return None
        sx, sy, _ = sites[facs[a]]
        px, py, w = points[i]
        total += w * math.hypot(px - sx, py - sy)
    return total


# =============================== references ==================================
def _cheapest_k(inst):
    order = sorted(range(inst["n_sites"]), key=lambda s: (inst["sites"][s][2], s))
    return order[:inst["k"]]


def _farthest_point_sampling(inst):
    """Classic k-center greedy: repeatedly add the candidate site maximizing the
    minimum distance to already-chosen sites. Geometry-only, weight/cost-blind,
    and by construction it spreads facilities apart -- it will always finish
    covering every macro district once before ever placing a SECOND facility
    inside the same district, even when a dense pocket there badly needs one."""
    sites = inst["sites"]; M = inst["n_sites"]; k = inst["k"]
    first = min(range(M), key=lambda s: sites[s][2])
    chosen = [first]
    mind = [math.hypot(sites[s][0] - sites[first][0], sites[s][1] - sites[first][1]) for s in range(M)]
    while len(chosen) < k:
        best = None; bestd = -1.0
        for s in range(M):
            if s in chosen:
                continue
            if mind[s] > bestd or (mind[s] == bestd and (best is None or s < best)):
                bestd = mind[s]; best = s
        chosen.append(best)
        for s in range(M):
            dd = math.hypot(sites[s][0] - sites[best][0], sites[s][1] - sites[best][1])
            if dd < mind[s]:
                mind[s] = dd
    return chosen


def _knn_scale(points, kk=5):
    """Demand-cluster diagnosis: local density scale per household, via distance to
    its kk-th nearest neighbor. Small -> dense pocket; large -> sparse/noise."""
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


def _adaptive_lns(inst, iters, alpha, rmin, rmax, pool_cap, seed, restarts):
    """Internal high-effort reference: density-scaled destroy/reinsert with a
    threshold-acceptance schedule, run from multiple restarts. Only more search
    budget than a solver gets inside its own time limit -- no hidden information."""
    sites = inst["sites"]; points = inst["points"]; M = inst["n_sites"]; k = inst["k"]
    scale = _knn_scale(points, kk=5)
    best_cost = None; best_facs = None
    rng = random.Random(seed)
    for r in range(restarts):
        if r == 0:
            cur = list(_farthest_point_sampling(inst))
        else:
            cur = rng.sample(range(M), k)
        cur_cost, cur_assign = _nearest_cost(inst, cur)
        run_best_cost, run_best_facs = cur_cost, list(cur)
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
            radius = min(max(alpha * scale[hot], rmin), rmax)
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
                avail = [s for s in pool if s not in chosen]
                if not avail:
                    avail = [s for s in not_open if s not in chosen]
                bestc, bests = None, None
                for cand in avail:
                    trial = remaining + chosen + [cand]
                    c, _ = _nearest_cost(inst, trial)
                    if bestc is None or c < bestc:
                        bestc, bests = c, cand
                chosen.append(bests)
            new_facs = remaining + chosen
            new_cost, new_assign = _nearest_cost(inst, new_facs)
            if new_cost <= cur_cost + thr:
                cur, cur_cost, cur_assign = new_facs, new_cost, new_assign
                if cur_cost < run_best_cost:
                    run_best_cost, run_best_facs = cur_cost, list(cur)
        if best_cost is None or run_best_cost < best_cost:
            best_cost, best_facs = run_best_cost, run_best_facs
    return best_cost, best_facs


# =============================== scoring driver ==============================
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        base_facs = _cheapest_k(inst)
        q_base, _ = _nearest_cost(inst, base_facs)
        q_ref, _ = _adaptive_lns(inst, iters=140, alpha=2.5, rmin=15.0, rmax=420.0,
                                  pool_cap=7, seed=inst["k"] * 97 + inst["n_demand"], restarts=3)
        denom = q_base - q_ref
        if denom < 1e-9:
            denom = 1e-9

        public = {
            "name": inst["name"], "n_demand": inst["n_demand"], "n_sites": inst["n_sites"],
            "k": inst["k"], "points": [list(p) for p in inst["points"]],
            "sites": [list(s) for s in inst["sites"]],
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _score_answer(inst, ans)
        except Exception:
            q_cand = None
        if q_cand is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (q_base - q_cand) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
