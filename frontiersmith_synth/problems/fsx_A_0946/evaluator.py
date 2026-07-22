#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0946 -- "Plumbing the Village Above the Permafrost"
(family: frost-front-pipe-routing; format B, quality-metric).

THEME.  A village of K buildings sits on a checkerboard of ground-cover types
(bare soil, plowed pavement, snow-holding open field, insulating peat/bog).
Every winter the frost front drives down from the surface; how deep it gets
by season's end depends on the CELL'S cover type and on THAT winter's cold
and snowfall -- not on distance from any building.  The candidate must route
ONE connected water-pipe network linking all K buildings across the grid and
choose a burial depth for every cell the pipe passes through. Excavation
cost is QUADRATIC in depth (deeper trenches cost much more per metre), so
the true per-cell cost is soil-dependent and highly non-uniform: a cell's
"weight" in the true metric is required_depth(cover_type)**2, not 1. Because
that weight varies ~4x between cover types, the metrically-shortest network
is very often NOT the geometrically-shortest one -- a longer route threaded
through protective peat/snow terrain can cost far less to bury than the
short route across exposed pavement. This is the innovation hook: route and
depth must be jointly optimized in this warped, soil-dependent cost space.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
    {"name": str, "rows": R, "cols": C,
     "grid": [[0|1|2|3, ... C ...], ... R ...],   # cover type per cell
     "buildings": [[r,c], ...],                    # K entries, must all connect
     "winters": [{"fdd": float, "snow": float}, ...],  # historical winter record
     "kappa": {"bare":.., "pavement":.., "snow_base":.., "snow_sensitivity":.., "peat":..},
     "depth_scale": float,
     "type_names": {"0":"bare","1":"pavement","2":"snow_holding","3":"peat"}}
  stdout: ONE JSON object:
    {"route": [{"r":int, "c":int, "depth":float}, ...]}   # every cell the pipe
                                                            # occupies + its burial depth

FROST PHYSICS (evaluator-executed, stated honestly -- the candidate can and
should reproduce it exactly; nothing about the mechanism is hidden, only
which winter turns out worst for which cover type is left for the solver to
discover). For cover type t and winter w:
    kappa_eff(bare, w)     = kappa.bare
    kappa_eff(pavement, w) = kappa.pavement
    kappa_eff(snow, w)     = kappa.snow_base / (1 + kappa.snow_sensitivity * snow_w)
    kappa_eff(peat, w)     = kappa.peat
    frost_depth(t, w) = depth_scale * sqrt(2 * kappa_eff(t, w) * fdd_w)
    required_depth(t) = max over ALL winters w of frost_depth(t, w)
A pipe cell of type t buried shallower than required_depth(t) freezes solid
in at least one recorded winter -- catastrophic failure, not a graded miss.
Because snow-holding ground's insulation depends on that winter's SNOWFALL
(not its cold), the worst winter for a snow-holding cell is often NOT the
coldest winter on record -- unlike bare soil, pavement, and peat, whose
worst winter is always simply the one with the highest freezing-degree-days.
A policy that assumes "the coldest winter is worst for every cover type"
mis-sizes exactly the snow-holding cells.

SCORING (deterministic; no wall-time). Total excavation cost = sum of
depth**2 over every cell in the candidate's route (route must be a single
4-connected component containing all K buildings; any freeze, disconnection,
missing building, duplicate cell, or non-finite/out-of-range depth makes the
instance score 0.0). The evaluator computes, itself, two references never
sent to the candidate: cost_naive (a straight-line chain route buried at one
uniform, deliberately over-conservative depth -- the "safe and lazy" recipe)
and cost_ref (a strong internal lower reference: the best of K runs of a
warped-metric nearest-terminal Dijkstra Steiner heuristic, scaled down for
headroom). The per-instance score is
    r = clamp(0.1 + 0.9*(cost_naive - cost_cand)/max(1, cost_naive-cost_ref), 0, 1)
Matching the naive recipe scores ~0.1; a smarter but still geometry-blind
route scores clearly higher; genuinely warped-metric routing scores higher
still. Final Ratio is the mean of the 10 instances.

ISOLATION. The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the PUBLIC instance above. Every
reference and all validation happen in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean of the 10 instances' scores, in [0,1]>
  Vector: [r_1, ..., r_10]
"""
import sys, json, math, heapq
import isorun

N_INSTANCES = 10
NUM_WINTERS = 10
TYPES = (0, 1, 2, 3)   # 0=bare 1=pavement 2=snow_holding 3=peat
TYPE_NAMES = {0: "bare", 1: "pavement", 2: "snow_holding", 3: "peat"}

KAPPA = {"bare": 1.00, "pavement": 1.55, "snow_base": 1.00,
         "snow_sensitivity": 0.9, "peat": 0.38}
DEPTH_SCALE = 0.03
FDD_LO, FDD_HI = 800.0, 2200.0
SNOW_LO, SNOW_HI = 0.0, 2.5
NAIVE_MARGIN = 1.5     # trivial/naive recipe over-buries by this factor
REF_MARGIN = 0.55       # scales the internal "best-of-starts" reference down for headroom

MAX_DEPTH = 50.0


# ----------------------------- deterministic RNG ----------------------------
def _rng(seed):
    state = seed & ((1 << 64) - 1)

    def nxt():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return state

    def nxt_float():
        return (nxt() >> 11) / float(1 << 53)

    def nxt_int(lo, hi):
        return lo + (nxt() >> 17) % (hi - lo + 1)

    return nxt_float, nxt_int


# ----------------------------- frost physics ---------------------------------
def _kappa_eff(t, snow_w):
    if t == 0:
        return KAPPA["bare"]
    if t == 1:
        return KAPPA["pavement"]
    if t == 2:
        return KAPPA["snow_base"] / (1.0 + KAPPA["snow_sensitivity"] * snow_w)
    return KAPPA["peat"]


def _required_depths(winters):
    rd = {}
    for t in TYPES:
        best = 0.0
        for w in winters:
            k = _kappa_eff(t, w["snow"])
            d = DEPTH_SCALE * math.sqrt(2.0 * k * w["fdd"])
            if d > best:
                best = d
        rd[t] = best
    return rd


# ----------------------------- instance construction --------------------------
def _slots(R, C):
    m = 1
    return {
        "TL": (m, m), "TR": (m, C - 1 - m), "BL": (R - 1 - m, m), "BR": (R - 1 - m, C - 1 - m),
        "TM": (m, C // 2), "BM": (R - 1 - m, C // 2), "LM": (R // 2, m), "RM": (R // 2, C - 1 - m),
        "CT": (R // 2, C // 2),
    }


def _gen_terrain(seed, R, C, offset, horiz):
    nxt_float, nxt_int = _rng(seed)
    grid = [[1] * C for _ in range(R)]      # default: pavement (worst)
    for r in range(R):
        for c in range(C):
            if nxt_float() < 0.12:
                grid[r][c] = 0               # scattered bare-soil texture
    path = []
    if horiz:
        row = R // 2 + offset
        col = 0
        while col < C:
            for dr in (0, 1):
                rr = row + dr
                if 0 <= rr < R:
                    path.append((rr, col))
            row = max(1, min(R - 3, row + (nxt_int(0, 2) - 1)))
            col += 1
    else:
        col0 = C // 2 + offset
        row = 0
        while row < R:
            for dc in (0, 1):
                cc = col0 + dc
                if 0 <= cc < C:
                    path.append((row, cc))
            col0 = max(1, min(C - 3, col0 + (nxt_int(0, 2) - 1)))
            row += 1
    for i, (r, c) in enumerate(path):
        grid[r][c] = 3 if (i // 3) % 2 == 0 else 2   # alternating peat/snow corridor
    return grid


_CONFIGS = [
    dict(R=14, C=14, slots=["TL", "TR", "BL", "BR"], offset=-3, horiz=True),
    dict(R=14, C=14, slots=["TL", "TR", "BL", "BR"], offset=3, horiz=True),
    dict(R=13, C=15, slots=["TL", "BR", "RM"], offset=-2, horiz=False),
    dict(R=14, C=14, slots=["TL", "TR", "BL", "BR"], offset=2, horiz=True),
    dict(R=15, C=13, slots=["TL", "TR", "BL", "BR"], offset=0, horiz=False),
    dict(R=14, C=14, slots=["TL", "TR", "BL", "BR", "CT"], offset=-4, horiz=True),
    dict(R=14, C=14, slots=["TL", "TR", "BL", "BR"], offset=4, horiz=True),
    dict(R=16, C=16, slots=["TL", "TR", "BL", "BR", "CT"], offset=-2, horiz=False),
    dict(R=12, C=14, slots=["TL", "TR", "BL", "BR"], offset=1, horiz=True),
    dict(R=14, C=16, slots=["TL", "TR", "BL", "BR", "CT"], offset=0, horiz=True),
]


def _build_instance(idx):
    cfg = _CONFIGS[idx]
    seed = 90460 + idx * 977
    nxt_float, _ = _rng(seed)
    winters = []
    for _w in range(NUM_WINTERS):
        fdd = FDD_LO + nxt_float() * (FDD_HI - FDD_LO)
        snow = nxt_float() * (SNOW_HI - SNOW_LO) + SNOW_LO
        winters.append({"fdd": round(fdd, 4), "snow": round(snow, 4)})
    R, C = cfg["R"], cfg["C"]
    grid = _gen_terrain(seed + 2, R, C, cfg["offset"], cfg["horiz"])
    S = _slots(R, C)
    buildings = [list(S[name]) for name in cfg["slots"]]
    rd = _required_depths(winters)
    return {"name": f"village{idx:02d}", "rows": R, "cols": C, "grid": grid,
            "buildings": buildings, "winters": winters, "rd": rd}


def _build_instances():
    return [_build_instance(i) for i in range(N_INSTANCES)]


def _public_view(inst):
    return {
        "name": inst["name"], "rows": inst["rows"], "cols": inst["cols"],
        "grid": [list(row) for row in inst["grid"]],
        "buildings": [list(b) for b in inst["buildings"]],
        "winters": [dict(w) for w in inst["winters"]],
        "kappa": dict(KAPPA), "depth_scale": DEPTH_SCALE,
        "type_names": {str(k): v for k, v in TYPE_NAMES.items()},
    }


# ----------------------------- reference solvers ------------------------------
def _manhattan_chain_cost(inst):
    """Naive/lazy recipe: connect buildings in given order via straight L-paths,
    bury the ENTIRE route at one uniform, deliberately over-conservative depth
    (NAIVE_MARGIN times the worst required depth over ALL cover types)."""
    buildings = [tuple(b) for b in inst["buildings"]]
    route = {buildings[0]}
    for i in range(len(buildings) - 1):
        r0, c0 = buildings[i]
        r1, c1 = buildings[i + 1]
        rstep = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + rstep, rstep):
            route.add((r, c0))
        cstep = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + cstep, cstep):
            route.add((r1, c))
    ud = NAIVE_MARGIN * max(inst["rd"].values())
    return (ud * ud) * len(route)


def _weighted_steiner_cost(inst, start_idx):
    """Warped-metric nearest-terminal Dijkstra Steiner heuristic: repeatedly
    connect the cheapest-to-reach unconnected building to the current tree,
    where the cost to enter a NEW cell is required_depth(its type)**2 and
    entering a cell already in the tree is free."""
    R, C = inst["rows"], inst["cols"]
    grid = inst["grid"]
    rd = inst["rd"]
    buildings = [tuple(b) for b in inst["buildings"]]
    w = [[rd[grid[r][c]] ** 2 for c in range(C)] for r in range(R)]
    n = len(buildings)
    tree = {buildings[start_idx]}
    remaining = set(range(n)) - {start_idx}
    total = w[buildings[start_idx][0]][buildings[start_idx][1]]
    while remaining:
        dist = [[math.inf] * C for _ in range(R)]
        prev = {}
        pq = []
        for (r, c) in tree:
            dist[r][c] = 0.0
            heapq.heappush(pq, (0.0, r, c))
        while pq:
            d, r, c = heapq.heappop(pq)
            if d > dist[r][c]:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < R and 0 <= nc < C:
                    step = 0.0 if (nr, nc) in tree else w[nr][nc]
                    nd = d + step
                    if nd < dist[nr][nc] - 1e-12:
                        dist[nr][nc] = nd
                        prev[(nr, nc)] = (r, c)
                        heapq.heappush(pq, (nd, nr, nc))
        best_ri, best_d = None, math.inf
        for ri in remaining:
            b = buildings[ri]
            if dist[b[0]][b[1]] < best_d:
                best_d = dist[b[0]][b[1]]
                best_ri = ri
        cur = buildings[best_ri]
        path = [cur]
        while cur in prev:
            cur = prev[cur]
            path.append(cur)
        added = 0.0
        for cell in path:
            if cell not in tree:
                added += w[cell[0]][cell[1]]
                tree.add(cell)
        total += added
        remaining.discard(best_ri)
    return total


def _reference_cost(inst):
    """cost_ref: best-of-K-starts warped-metric Steiner cost, scaled down by
    REF_MARGIN so the shipped `strong` reference solution (a single fixed
    start, no exhaustive search) does not saturate the score."""
    best = min(_weighted_steiner_cost(inst, s) for s in range(len(inst["buildings"])))
    return best * REF_MARGIN


def naive_cost(inst):
    return _manhattan_chain_cost(inst)


# ----------------------------- answer validation ------------------------------
def _validate_and_score(inst, answer):
    if not isinstance(answer, dict):
        return False, None
    route = answer.get("route")
    if not isinstance(route, list) or not route or len(route) > inst["rows"] * inst["cols"]:
        return False, None
    R, C = inst["rows"], inst["cols"]
    grid = inst["grid"]
    rd = inst["rd"]
    cells = {}
    for item in route:
        if not isinstance(item, dict):
            return False, None
        r, c, d = item.get("r"), item.get("c"), item.get("depth")
        if isinstance(r, bool) or not isinstance(r, int):
            return False, None
        if isinstance(c, bool) or not isinstance(c, int):
            return False, None
        if not (0 <= r < R and 0 <= c < C):
            return False, None
        if isinstance(d, bool) or not isinstance(d, (int, float)):
            return False, None
        d = float(d)
        if d != d or d in (float("inf"), float("-inf")) or d < 0.0 or d > MAX_DEPTH:
            return False, None
        if (r, c) in cells:
            return False, None      # duplicate cell
        cells[(r, c)] = d

    # every building must be present
    for br, bc in inst["buildings"]:
        if (br, bc) not in cells:
            return False, None

    # single 4-connected component
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    if len(seen) != len(cells):
        return False, None

    # freeze check + cost
    eps = 1e-6
    cost = 0.0
    for (r, c), d in cells.items():
        req = rd[grid[r][c]]
        if d < req - eps:
            return False, None      # pipe freezes -> catastrophic failure
        cost += d * d
    return True, cost


# ----------------------------------- main ------------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = _public_view(inst)
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, cost = _validate_and_score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue

        cost_naive = naive_cost(inst)
        cost_ref = _reference_cost(inst)
        denom = cost_naive - cost_ref
        if denom < 1.0:
            denom = 1.0
        r = 0.1 + 0.9 * (cost_naive - cost) / denom
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        if not (r == r):
            r = 0.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
