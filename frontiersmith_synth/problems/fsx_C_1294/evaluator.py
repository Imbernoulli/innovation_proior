#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1294 -- "Cutting Trees That Have to Grow Back: A Multi-Year
Harvest Plan" (family: forest-harvest-policy; format B, quality-metric).

THEME.  A forest stand is an N x N grid.  Cell (r, c) holds an integer stage in
[0, S]: 0 means empty ground, 1..S-1 are growing trees, and S is a fully mature
"old growth" tree.  A tree cut at stage k is worth k*k (a stage-S tree is worth
S*S -- by far the most valuable single cut).  Every year:

  1. HARVEST.  The manager cuts at most `quota` distinct non-empty cells; each cut
     cell's value is banked and the cell becomes empty (stage 0).
  2. GROWTH.   Every remaining tree with 1 <= stage < S grows by exactly 1 stage.
  3. DISPERSAL. Every empty cell (stage 0, whether just cut or already bare) looks
     at the (2*radius+1) x (2*radius+1) Chebyshev-`radius` window around it (post-
     growth board).  If that window contains at least `min_seed` cells at stage S,
     a seedling establishes there (stage becomes 1).  Otherwise it stays bare.

CRITICALLY, `min_seed` >= 1 always and the maturity threshold for acting as a seed
source is exactly S -- the SAME stage that gives a tree its highest cutting value.
Cutting the biggest trees first maximizes each single year's yield, but it is also
the only way to destroy the region's seed sources: once every stage-S cell within
reach of a patch is gone, that patch can never regrow (dispersal requires an
existing stage-S neighbour; growth alone cannot create one from nothing).  A policy
that keeps a spatially spread subset of mature trees uncut -- harvesting AROUND
that network instead of through it -- keeps the whole stand producing indefinitely.

The manager submits one fixed plan for the WHOLE horizon up front (everything
needed to simulate forward -- the grid, the growth/dispersal/harvest rules, and all
public parameters -- is in the instance the candidate receives, and the dynamics
are fully deterministic given that plan, so there is no need for the candidate to
see intermediate years).  The manager MAXIMIZES total value cut over the horizon.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": N, "s_max": S, "radius": R, "min_seed": M,
             "quota": Q, "horizon": T, "grid": [[...], ...]}   # N rows of N ints
  stdout: ONE JSON object:
            {"harvests": [[[r,c], ...], ...]}   # length <= T; year t's list has
                                                 # length <= Q, distinct in-range
                                                 # cells that hold a tree (stage>=1)
                                                 # at the moment year t is played.
          Missing trailing years count as "cut nothing that year".  Any structural
          violation (bad shape/type, out-of-range or duplicate cell within a year,
          more than `quota` cells in a year, more than `horizon` years, or a cut
          claimed on a cell that is not actually a tree at that point in the
          replay) makes the WHOLE instance score 0.0.

SCORING (deterministic; no wall-time).  Per instance:
    y_triv = value of the evaluator's own weak reference plan: each year, cut the
             first `quota` non-empty cells found in row-major scan order, ignoring
             value and regrowth completely.
    y_ub   = quota * horizon * S*S.  A loose, generally unreachable upper bound (it
             assumes literally every single cut, every year, is a stage-S tree --
             impossible once growth/dispersal limits the mature-tree supply).
    y_cand = value of the candidate's (strictly validated) plan, replayed by THIS
             evaluator against the true dynamics.
  normalized with an affine anchor (weak reference -> 0.1, loose ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (y_cand - y_triv) / max(1e-9, y_ub - y_triv), 0, 1 )

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  All replay,
validation, and reference computation happen in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- instance family -----------------------------
def _build_grid(seed, n, S, R, style, dens):
    """Deterministic N x N initial grid, stages in [0, S]. Two generation
    styles: 'clustered' (a few far-apart aged patches -- a seed source lost in
    one patch cannot be rescued by another) and 'dispersed' (sparse, uniformly
    scattered mature trees)."""
    ni = _rng(seed)
    grid = [[0] * n for _ in range(n)]
    if style == "clustered":
        spacing = max(3, 3 * R + 2)  # cluster centers spaced far apart relative to radius
        centers = []
        r = spacing // 2
        while r < n:
            c = spacing // 2
            while c < n:
                jr = min(n - 1, max(0, r + ni(-1, 1)))
                jc = min(n - 1, max(0, c + ni(-1, 1)))
                centers.append((jr, jc))
                c += spacing
            r += spacing
        for (cr, cc) in centers:
            rad = ni(1, 2)
            for rr in range(max(0, cr - rad), min(n, cr + rad + 1)):
                for cc2 in range(max(0, cc - rad), min(n, cc + rad + 1)):
                    if abs(rr - cr) + abs(cc2 - cc) <= rad + 1:
                        grid[rr][cc2] = ni(1, S)  # a naturally-aged patch
    else:  # dispersed: sparse, uniformly scattered mature trees, empty elsewhere
        for rr in range(n):
            for cc2 in range(n):
                if ni(0, 9999) < int(dens * 10000):
                    grid[rr][cc2] = S
    return grid


def _build_instances():
    """Deterministic instance family: (seed, n, S, radius, min_seed, quota,
    horizon, style, dens). style/dens only affect grid generation."""
    specs = [
        ("stand01", 33, 10, 6, 1, 1, 3, 30, "dispersed", 0.05),
        ("stand02", 11, 10, 6, 1, 1, 3, 40, "dispersed", 0.05),
        ("stand03", 22, 10, 6, 1, 1, 10, 30, "clustered", 0.0),
        ("stand04", 22, 12, 6, 1, 2, 3, 30, "dispersed", 0.05),
        ("stand05", 11, 12, 6, 2, 1, 8, 30, "clustered", 0.0),
        ("stand06", 11, 12, 6, 2, 2, 10, 30, "clustered", 0.0),
        ("stand07", 22, 14, 6, 1, 2, 3, 30, "dispersed", 0.05),
        ("stand08", 11, 14, 6, 2, 1, 10, 30, "clustered", 0.0),
        ("stand09", 33, 14, 6, 2, 2, 15, 40, "clustered", 0.0),
        ("stand10", 22, 14, 6, 1, 2, 4, 30, "dispersed", 0.03),
    ]
    out = []
    for (name, seed, n, S, R, minseed, Q, T, style, dens) in specs:
        grid = _build_grid(seed, n, S, R, style, dens)
        out.append({"name": name, "n": n, "s_max": S, "radius": R,
                    "min_seed": minseed, "quota": Q, "horizon": T, "grid": grid})
    return out


# ----------------------------- world dynamics -------------------------------
def _neighbors_within(r, c, R, n):
    r0, r1 = max(0, r - R), min(n - 1, r + R)
    c0, c1 = max(0, c - R), min(n - 1, c + R)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if rr == r and cc == c:
                continue
            yield rr, cc


def _step_world(grid, n, R, S, minseed):
    """Growth then dispersal, in place."""
    for rr in range(n):
        for cc in range(n):
            if 1 <= grid[rr][cc] < S:
                grid[rr][cc] += 1
    newly = []
    for rr in range(n):
        for cc in range(n):
            if grid[rr][cc] == 0:
                cnt = 0
                for (nr, nc) in _neighbors_within(rr, cc, R, n):
                    if grid[nr][nc] >= S:
                        cnt += 1
                        if cnt >= minseed:
                            break
                if cnt >= minseed:
                    newly.append((rr, cc))
    for (rr, cc) in newly:
        grid[rr][cc] = 1


def _value(stage):
    return stage * stage


def _trivial_plan_yield(grid0, n, R, S, minseed, Q, T):
    """Weak reference: row-major FIFO harvest of harvestable cells, ignoring value."""
    grid = [row[:] for row in grid0]
    total = 0.0
    for _t in range(T):
        picks = []
        for r in range(n):
            for c in range(n):
                if len(picks) >= Q:
                    break
                if grid[r][c] >= 1:
                    picks.append((r, c))
            if len(picks) >= Q:
                break
        for (r, c) in picks:
            total += _value(grid[r][c])
            grid[r][c] = 0
        _step_world(grid, n, R, S, minseed)
    return total


def _validate_and_replay(inst, answer):
    """Strictly validate `answer` against `inst` and replay it against the true
    dynamics. Returns (ok: bool, total_value: float)."""
    if not isinstance(answer, dict):
        return False, 0.0
    harvests = answer.get("harvests")
    if not isinstance(harvests, list):
        return False, 0.0
    n = inst["n"]; R = inst["radius"]; S = inst["s_max"]
    minseed = inst["min_seed"]; Q = inst["quota"]; T = inst["horizon"]
    if len(harvests) > T:
        return False, 0.0
    grid = [row[:] for row in inst["grid"]]
    total = 0.0
    for t in range(T):
        picks_raw = harvests[t] if t < len(harvests) else []
        if not isinstance(picks_raw, list) or len(picks_raw) > Q:
            return False, 0.0
        seen = set()
        year_value = 0.0
        for pair in picks_raw:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                return False, 0.0
            r, c = pair[0], pair[1]
            if isinstance(r, bool) or isinstance(c, bool):
                return False, 0.0
            if not isinstance(r, int) or not isinstance(c, int):
                return False, 0.0
            if r < 0 or r >= n or c < 0 or c >= n:
                return False, 0.0
            if (r, c) in seen:
                return False, 0.0
            seen.add((r, c))
            if grid[r][c] < 1:
                return False, 0.0
            year_value += _value(grid[r][c])
            grid[r][c] = 0
        total += year_value
        _step_world(grid, n, R, S, minseed)
    return True, total


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        n = inst["n"]; R = inst["radius"]; S = inst["s_max"]
        minseed = inst["min_seed"]; Q = inst["quota"]; T = inst["horizon"]
        y_triv = _trivial_plan_yield(inst["grid"], n, R, S, minseed, Q, T)
        y_ub = Q * T * _value(S)
        denom = y_ub - y_triv
        if denom < 1e-9:
            denom = 1e-9

        public = {"name": inst["name"], "n": n, "s_max": S, "radius": R,
                  "min_seed": minseed, "quota": Q, "horizon": T,
                  "grid": [list(row) for row in inst["grid"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, y_cand = _validate_and_replay(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue

        r = 0.1 + 0.9 * (y_cand - y_triv) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
