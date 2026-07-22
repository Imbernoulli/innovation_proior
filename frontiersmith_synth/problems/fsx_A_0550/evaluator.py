#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0550 -- "Firebreak Dispatch on the Advancing Front"
(family: propagating-frontier-dispatch; format B, quality-metric).

THEME.  A wildfire ignites on a rectangular cell grid and spreads DETERMINISTICALLY
outward, funnelled by wind and fuel.  You command a small crew of fire teams.  Each
team you dispatch builds a FIREBREAK at one cell -- but the build takes `build_time`
steps, during which that team is LOCKED (crew-commitment lock-in): it cannot be
redeployed until the firebreak is finished.  A firebreak only stops the fire if it is
COMPLETED strictly before the flame front reaches that cell; a break whose cell ignites
mid-build is overrun and does nothing.  You have `crews` teams, so at most `crews`
firebreaks may be under construction at the same instant.  The score is the total
`value` of grid area still UNBURNED at the horizon.  The trap: hurling teams at the
cells the fire is touching right now wastes them -- those cells are locked while the
front sweeps around and past them.  The win is anticipation: read where wind + fuel
will carry the front, and pre-build a break on that FUTURE frontier so it stands
finished the moment the fire arrives, sealing off the ground behind it.

DETERMINISTIC FIRE MODEL (both the evaluator and any solver reproduce this exactly).
  Cells are 4-connected.  Each cell v has integer fuel-resistance res[v] >= 1.  Fire
  spreads from a burning cell u to an orthogonal neighbour v with an integer STEP COST
      cost(u->v) = max(1, res[v] + wind_strength * align(u->v)),
  where dir = (v.r - u.r, v.c - u.c) is the unit step and
      align(u->v) = -(dir . wind).
  Moving DOWN-wind (dir aligned with `wind`) subtracts wind_strength (fire runs fast);
  moving UP-wind adds it (fire crawls).  Seed cells ignite at time 0.  The ignition
  ("arrival") time of every other cell is the minimum total step cost of any path from
  a seed, i.e. a shortest-path / weighted-BFS field.  A cell BURNS at its arrival time;
  a cell whose arrival time exceeds `horizon` never burns within the episode.

  A firebreak placed at cell x with dispatch time d completes at time d + build_time.
  When the front's arrival time A(x) satisfies (d + build_time) <= A(x), cell x becomes
  a permanent barrier: it never burns and the fire cannot spread THROUGH it.  Otherwise
  the break is overrun (arrival <= completion) and cell x behaves as ordinary fuel.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": N,
             "res":   [[...], ...],   # N x N ints >= 1 (fuel resistance / step cost)
             "value": [[...], ...],   # N x N ints >= 0 (asset value of the cell)
             "seeds": [[r,c], ...],   # ignition cells (arrival time 0)
             "wind":  [wr, wc],       # each component in {-1,0,1}
             "wind_strength": ws,     # int >= 0
             "build_time": b,         # steps to complete a firebreak
             "crews": C,              # max simultaneous builds (lock-in capacity)
             "horizon": T}            # scoring horizon (steps)
  stdout: ONE JSON object:
            {"breaks": [[r, c, d], ...]}     # firebreak at (r,c) dispatched at step d
          Each entry: 0 <= r,c < N integers, d >= 0 integer.  Cells must be DISTINCT.
          FEASIBILITY (lock-in): for every instant t, the number of breaks whose build
          interval [d, d+b) contains t must be <= C.  Any violation, out-of-range cell,
          duplicate cell, non-integer field, crash, timeout, or non-JSON -> instance 0.0.
          An EMPTY list is feasible (dispatch nobody) and scores the do-nothing baseline.

SCORING (deterministic; no wall-time).  Per instance:
    saved(breaks) = total value of cells NOT burned by the horizon under the model above.
    y_base = saved([])                 # do-nothing: the weak baseline -> normalized 0.1
    y_ub   = sum of ALL cell values    # save everything (unreachable: seeds always burn)
    y_cand = saved(candidate breaks)
  normalized with an affine anchor (do-nothing -> 0.1, save-everything -> 1.0):
    r = clamp( 0.1 + 0.9 * (y_cand - y_base) / max(1e-9, y_ub - y_base), 0, 1 )
  Matching do-nothing scores ~0.1; sealing more valuable ground scores higher; because
  the ground near the seed always burns and crews are scarce, even a strong siter stays
  well below 1.0 -> open headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  All references and
validation happen in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, heapq
import isorun

RIDGE = 60  # fuel-resistance of impassable ridge rock (>> any horizon -> natural barrier)


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


# ----------------------------- instance family -----------------------------
# Each spec builds a map with a vertical ridge (natural barrier) pierced by one or two
# narrow GAPS.  The fire is seeded on the left, wind drives it right toward the ridge,
# it funnels through the gap(s), and a high-value TOWN sits behind a gap.  Sealing the
# gap on the FUTURE frontier saves the town; walling the flame edge near the seed does
# not.  Two-gap specs plant a value trade-off (town gap vs. lower-value hamlet gap) that
# scarce crews cannot both seal.
def _spec_list():
    # (seed, n, seed_row, wind, ws, ridge_col, gaps, town, hamlet, b, crews, T)
    #   gaps   = list of (gap_row_start, width)   at column ridge_col
    #   town   = (r0, r1, c0off, c1off, vlo, vhi) rows [r0,r1), cols [ridge_col+c0off, +c1off)
    #   hamlet = same tuple or None
    return [
        # ---- single-gap trap cases (flame-edge greedy misses the far chokepoint) ----
        (1101, 24, 11, (0, 1), 2, 12, [(10, 2)], (7, 17, 2, 11, 40, 80), None, 6, 3, 46),
        (1102, 24, 12, (0, 1), 3, 12, [(11, 2)], (6, 18, 2, 11, 45, 85), None, 5, 3, 44),
        (1103, 26, 13, (0, 1), 2, 13, [(12, 3)], (8, 18, 2, 12, 40, 75), None, 7, 3, 50),
        (1104, 22, 10, (0, 1), 2, 11, [(9, 2)],  (5, 16, 2, 10, 50, 90), None, 6, 3, 44),
        (1105, 28, 14, (0, 1), 3, 14, [(13, 3)], (9, 20, 2, 13, 40, 80), None, 6, 4, 54),
        # ---- diagonal wind: future frontier drifts, greedy's straight wall misses ----
        (1106, 26, 9,  (1, 1), 2, 13, [(15, 2)], (12, 22, 2, 12, 45, 85), None, 6, 3, 52),
        (1107, 26, 16, (-1, 1), 2, 13, [(9, 2)], (4, 14, 2, 12, 45, 85), None, 6, 3, 52),
        # ---- two-gap trade-off: crews can seal only one; strong picks the town ----
        (1108, 28, 14, (0, 1), 2, 14, [(7, 2), (19, 2)], (5, 12, 2, 13, 55, 95),
         (17, 24, 2, 13, 12, 26), 6, 3, 54),
        (1109, 30, 15, (0, 1), 3, 15, [(8, 2), (21, 2)], (18, 26, 2, 14, 55, 95),
         (6, 14, 2, 14, 12, 26), 6, 3, 58),
        # ---- wider gap, more crews: still must anticipate, not chase the edge ----
        (1110, 30, 15, (0, 1), 2, 15, [(13, 4)], (9, 22, 2, 14, 40, 80), None, 7, 4, 58),
    ]


def _build(spec):
    (seed, n, seed_row, wind, ws, ridge_col, gaps, town, hamlet, b, crews, T) = spec
    rng = _rng(seed)
    res = [[0] * n for _ in range(n)]
    val = [[0] * n for _ in range(n)]
    # base fuel + low wildland value everywhere
    for r in range(n):
        for c in range(n):
            res[r][c] = rng(2, 4)
            val[r][c] = rng(1, 3)
    # left of the ridge: a populated valley (moderate value) that the fire starts in
    # and overruns early -- largely un-saveable, it keeps the score ceiling open.
    for r in range(n):
        for c in range(1, ridge_col):
            val[r][c] = rng(6, 16)
    # ridge column: impassable rock (except at the gaps)
    gap_rows = set()
    for (gr, w) in gaps:
        for k in range(w):
            gap_rows.add(gr + k)
    for r in range(n):
        if r in gap_rows:
            res[r][ridge_col] = 2      # the passage: ordinary fuel
            val[r][ridge_col] = rng(1, 3)
        else:
            res[r][ridge_col] = RIDGE  # rock
            val[r][ridge_col] = 0
    # town (high value) and optional hamlet (medium value), behind the ridge
    def paint(region):
        (r0, r1, c0off, c1off, vlo, vhi) = region
        for r in range(max(0, r0), min(n, r1)):
            for c in range(ridge_col + c0off, min(n, ridge_col + c1off)):
                val[r][c] = rng(vlo, vhi)
    paint(town)
    if hamlet is not None:
        paint(hamlet)
    seeds = [[seed_row, 1]]
    public = {"name": "burn%d" % seed, "n": n, "res": res, "value": val,
              "seeds": seeds, "wind": [wind[0], wind[1]], "wind_strength": ws,
              "build_time": b, "crews": crews, "horizon": T}
    return public


# ----------------------------- deterministic fire model --------------------
def _saved_value(pub, breaks):
    """Total value of cells NOT burned by the horizon, given firebreak list `breaks`
    (each [r,c,d]).  Event-driven weighted-BFS: a break at x is a barrier iff it is
    completed (d+b) no later than x's arrival time."""
    n = pub["n"]; res = pub["res"]; val = pub["value"]
    wr, wc = pub["wind"]; ws = pub["wind_strength"]
    b = pub["build_time"]; T = pub["horizon"]
    comp = {}
    for (r, c, d) in breaks:
        comp[r * n + c] = d + b
    INF = float("inf")
    arrival = [INF] * (n * n)
    done = [False] * (n * n)
    pq = []
    for (sr, sc) in pub["seeds"]:
        idx = sr * n + sc
        if arrival[idx] > 0:
            arrival[idx] = 0
            heapq.heappush(pq, (0, idx))
    deficit = 0.0                      # value lost to burning, weighted by earliness
    Tf = float(T)
    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while pq:
        a, idx = heapq.heappop(pq)
        if done[idx]:
            continue
        if a > T:
            break                      # everything left survives the horizon (full value)
        done[idx] = True
        ct = comp.get(idx)
        if ct is not None and ct <= a:
            continue                   # firebreak finished in time -> barrier, survives fully
        r, c = divmod(idx, n)
        # a cell that burns at time a keeps only the fraction a/T of its value
        # (survived time); an unburned/protected cell keeps all of it.
        deficit += val[r][c] * (1.0 - a / Tf)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n:
                nidx = nr * n + nc
                if done[nidx]:
                    continue
                align = -(dr * wr + dc * wc)
                cost = res[nr][nc] + ws * align
                if cost < 1:
                    cost = 1
                na = a + cost
                if na < arrival[nidx]:
                    arrival[nidx] = na
                    heapq.heappush(pq, (na, nidx))
    total = sum(sum(row) for row in val)
    return total - deficit, total


# ----------------------------- validation ----------------------------------
def _validate(pub, answer):
    """Return a list of [r,c,d] breaks, or None if infeasible."""
    if not isinstance(answer, dict):
        return None
    breaks = answer.get("breaks")
    if breaks is None:
        return None
    if not isinstance(breaks, list):
        return None
    n = pub["n"]; b = pub["build_time"]; C = pub["crews"]
    seen = set()
    out = []
    starts = []
    for e in breaks:
        if not isinstance(e, (list, tuple)) or len(e) != 3:
            return None
        r, cc, d = e[0], e[1], e[2]
        for v in (r, cc, d):
            if isinstance(v, bool) or not isinstance(v, int):
                return None
        if r < 0 or r >= n or cc < 0 or cc >= n or d < 0:
            return None
        if (r, cc) in seen:
            return None
        seen.add((r, cc))
        out.append([r, cc, d])
        starts.append(d)
    # lock-in capacity: at most C build intervals [d, d+b) overlap any instant.
    # equal-or-mixed length intervals -> max overlap == max clique; check at each start.
    if b <= 0:
        return None
    for s in starts:
        active = 0
        for d in starts:
            if d <= s < d + b:
                active += 1
        if active > C:
            return None
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    specs = _spec_list()
    vec = []
    for spec in specs:
        pub = _build(spec)
        y_base, y_ub = _saved_value(pub, [])
        denom = y_ub - y_base
        if denom < 1e-9:
            denom = 1e-9
        ans, st = isorun.run_candidate(cand, pub, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            breaks = _validate(pub, ans)
        except Exception:
            breaks = None
        if breaks is None:
            vec.append(0.0)
            continue
        try:
            y_cand, _ = _saved_value(pub, breaks)
        except Exception:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (y_cand - y_base) / denom
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
