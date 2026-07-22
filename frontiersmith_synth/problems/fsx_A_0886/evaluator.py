#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0886 -- "Firebreaks Against an Unpredictable Blaze"
(family: percolation-firebreak-placement; format B, quality-metric).

THEME. A ranger has a flammability map of a forest grid (some cells already
bare/rock = non-flammable). A limited BUDGET of cells can be converted to
firebreak (removed from the flammable graph) BEFORE the season starts, with
no knowledge of exactly where or when a blaze will ignite. A historical
"hot zone" (the cell block that has ignited most often in the past) is
flagged, together with the season's prevailing wind direction, as PUBLIC
hints -- but real ignitions and real wind vary: a fixed fraction light up
uniformly ANYWHERE in the flammable footprint, and even when the fire does
start in the hot zone, the wind that carries it is only the "prevailing"
direction about half the time.

MECHANISMS composed:
  - budget-placement:        a hard cardinality budget K on firebreak cells,
                              chosen once, open-loop, before any ignition.
  - percolation-front:       fire spreads over the surviving flammable graph
                              via DIRECTED bond percolation -- each candidate
                              edge is open independently at a probability that
                              depends on its orientation relative to that
                              draw's wind (higher downwind, lower upwind,
                              baseline perpendicular). This is a genuine
                              percolation process, not a fixed "spread N
                              cells" rule.
  - critical-cluster-cutting: because the flammable footprint sits ABOVE the
                              percolation threshold, it forms one giant
                              connected cluster; a fire ANYWHERE in that
                              cluster can, in principle, reach almost all of
                              it. Expected damage is governed by the SIZE of
                              the piece the giant cluster gets fragmented
                              into by the firebreak placement, not by which
                              single corridor is blocked.

INNOVATION HOOK (what the strong solution must exploit): damage is controlled
by GLOBAL CONNECTIVITY. Spending the whole budget walling off (or blocking
the path out of) the historical hot zone only helps the minority of
ignitions that (a) start there AND (b) travel in the direction the wall
covers -- the rest of the mixture ignites elsewhere in the still-fully-
connected giant cluster and burns nearly all of it. A budget spent instead
FRAGMENTING the giant cluster into balanced pieces caps the expected burned
fraction near (fragment size / total), independent of where the fire starts
or which way the wind blows that day.

CANDIDATE CONTRACT (isolated stdin -> stdout program, called ONCE per
instance -- single-shot, open-loop placement; no interaction).

  stdin: ONE JSON public instance
    { "name": str, "R": int, "C": int,
      "flammable": [[0/1, ...], ...],      # R x C, 1 = flammable forest cell
      "budget": int,                       # max cells the answer may place
      "hint_zone": {"r0":int,"r1":int,"c0":int,"c1":int},   # historical hot
                     # zone (inclusive bounds); ALWAYS fully flammable
      "wind_bias": [dr, dc],               # this season's prevailing wind
                     # direction, one of (-1,0)/(1,0)/(0,-1)/(0,1)
      "p_base": float, "wind_extra": float,   # directed percolation open
                     # probs: downwind = p_base+wind_extra, upwind =
                     # p_base-wind_extra, perpendicular = p_base
      "p_wind_dominant": float,            # P(a given ignition's wind ==
                     # wind_bias); else uniform over the other 3 directions
      "w_hint": float,                     # P(a given ignition lands inside
                     # hint_zone); else uniform over ALL flammable cells
      "n_draws": int }                     # number of hidden ignition/wind
                     # draws the score is averaged over (realizations hidden)

  stdout: ONE JSON object   { "cells": [[r,c], [r,c], ...] }
    - Up to `budget` DISTINCT in-bounds cells, each with flammable[r][c]==1
      in the ORIGINAL map (you may not "place a firebreak" on ground that
      was never flammable). Any malformed / duplicate / out-of-bounds /
      out-of-budget / off-flammable-ground answer, crash, timeout, or
      non-JSON reply scores the whole instance 0.0.

SCORING (deterministic; no wall-time; nothing about the real ignitions/wind
realizations is ever sent to the candidate). Per instance the evaluator:
  1. Removes the answer's cells from the flammable graph.
  2. Replays `n_draws` HIDDEN ignition/wind scenarios (deterministically
     regenerated from the instance's internal seed -- same draws every run,
     same draws used for the baseline below, so the comparison is exact) --
     for each draw: an ignition cell (mixture of hint_zone / uniform-anywhere
     per `w_hint`) and a wind direction (mixture of wind_bias / uniform-other
     per `p_wind_dominant`), then a directed-percolation BFS flood over the
     surviving flammable graph using per-directed-edge open probabilities
     as above (each edge's open/closed draw is a deterministic hash of
     (instance seed, draw index, edge endpoints) -- reproducible regardless
     of BFS traversal order).
  3. obj = mean, over the n_draws scenarios, of (cells burned / total
     ORIGINALLY-flammable cell count). This is what the candidate minimizes.
  4. b = baseline(inst) = the SAME quantity with an EMPTY firebreak
     placement (do nothing), computed directly by the evaluator over the
     identical draws.
  5. r = min(1.0, 0.1 * b / max(obj, 1e-12)).  Doing nothing reproduces the
     baseline exactly -> r = 0.1. Every relative reduction in burned
     fraction below baseline raises r; no reduction (or making it worse)
     keeps r <= 0.1.

ISOLATION. The candidate is invoked exactly once via isorun.run_candidate
in a fresh OS-sandboxed subprocess; it sees only the public instance above.
The hidden per-scenario ignition/wind realizations and all RNG state live
only in this parent process.

CLI: python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
STEP_TIMEOUT = 15


# ----------------------------- deterministic hash RNG -----------------------
def _mix64(x):
    x &= (1 << 64) - 1
    x = (x ^ (x >> 33)) * 0xff51afd7ed558ccd & ((1 << 64) - 1)
    x = (x ^ (x >> 33)) * 0xc4ceb9fe1a85ec53 & ((1 << 64) - 1)
    x ^= (x >> 33)
    return x & ((1 << 64) - 1)


def _rand01(*parts):
    x = 0
    for p in parts:
        x = _mix64((x + (p & ((1 << 64) - 1)) + 0x9E3779B97F4A7C15) & ((1 << 64) - 1))
    return (x >> 11) / float(1 << 53)


# ----------------------------- grid / instance generation -------------------
def _gen_grid(seed, R, C, density, hint_zone):
    grid = [[0] * C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            v = _rand01(seed, 1000003 * r + 7, 2000003 * c + 11)
            grid[r][c] = 1 if v < density else 0
    r0, r1, c0, c1 = hint_zone["r0"], hint_zone["r1"], hint_zone["c0"], hint_zone["c1"]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            grid[r][c] = 1
    return grid


def _build_instances():
    """Deterministic instance family (10 boards). `trap=True` boards are
    engineered so that fully sealing the historical hot zone (or blocking
    its single most-likely downwind path) leaves the REST of the giant
    flammable cluster untouched, while a fragmenting cut caps damage
    everywhere regardless of where ignition / wind land that day."""
    specs = [
        # ---- TRAP boards: hint zone embedded WELL INSIDE one giant cluster;
        # w_hint well under 1 so most mass ignites elsewhere in that cluster ----
        dict(name="grove01", seed=8801, R=14, C=18, density=0.80, budget=14,
             hint_zone=dict(r0=2, r1=4, c0=2, c1=4), wind_bias=(0, 1),
             p_base=0.75, wind_extra=0.20, p_wind_dominant=0.5, w_hint=0.40, trap=True),
        dict(name="grove02", seed=8802, R=15, C=17, density=0.78, budget=15,
             hint_zone=dict(r0=10, r1=12, c0=11, c1=13), wind_bias=(1, 0),
             p_base=0.74, wind_extra=0.22, p_wind_dominant=0.5, w_hint=0.40, trap=True),
        dict(name="grove03", seed=8803, R=16, C=16, density=0.79, budget=15,
             hint_zone=dict(r0=6, r1=8, c0=1, c1=3), wind_bias=(0, -1),
             p_base=0.75, wind_extra=0.20, p_wind_dominant=0.55, w_hint=0.35, trap=True),
        dict(name="grove04", seed=8804, R=14, C=20, density=0.80, budget=16,
             hint_zone=dict(r0=3, r1=5, c0=15, c1=17), wind_bias=(-1, 0),
             p_base=0.76, wind_extra=0.20, p_wind_dominant=0.5, w_hint=0.40, trap=True),
        # ---- held-out TRAP boards, larger / different aspect ----
        dict(name="grove05", seed=8805, R=17, C=19, density=0.79, budget=18,
             hint_zone=dict(r0=7, r1=9, c0=2, c1=4), wind_bias=(0, 1),
             p_base=0.75, wind_extra=0.21, p_wind_dominant=0.5, w_hint=0.35, trap=True),
        # ---- PLAIN boards: still above the percolation threshold (one giant
        # cluster survives) but with a HIGHER w_hint, so the hot zone genuinely
        # captures most of the ignition mass and the trap effect is milder --
        # geometry / density still vary for generalization ----
        dict(name="grove06", seed=8806, R=13, C=15, density=0.77, budget=12,
             hint_zone=dict(r0=5, r1=6, c0=6, c1=7), wind_bias=(1, 0),
             p_base=0.72, wind_extra=0.18, p_wind_dominant=0.5, w_hint=0.65, trap=False),
        dict(name="grove07", seed=8807, R=14, C=14, density=0.78, budget=13,
             hint_zone=dict(r0=1, r1=2, c0=6, c1=7), wind_bias=(0, -1),
             p_base=0.72, wind_extra=0.18, p_wind_dominant=0.5, w_hint=0.60, trap=False),
        dict(name="grove08", seed=8808, R=15, C=13, density=0.78, budget=13,
             hint_zone=dict(r0=11, r1=12, c0=5, c1=6), wind_bias=(-1, 0),
             p_base=0.73, wind_extra=0.19, p_wind_dominant=0.5, w_hint=0.55, trap=False),
        dict(name="grove09", seed=8809, R=16, C=14, density=0.77, budget=13,
             hint_zone=dict(r0=7, r1=8, c0=6, c1=7), wind_bias=(0, 1),
             p_base=0.71, wind_extra=0.17, p_wind_dominant=0.5, w_hint=0.55, trap=False),
        # ---- held-out PLAIN, larger board ----
        dict(name="grove10", seed=8810, R=18, C=16, density=0.77, budget=16,
             hint_zone=dict(r0=8, r1=9, c0=7, c1=8), wind_bias=(1, 0),
             p_base=0.72, wind_extra=0.18, p_wind_dominant=0.5, w_hint=0.50, trap=False),
    ]
    out = []
    for spec in specs:
        grid = _gen_grid(spec["seed"], spec["R"], spec["C"], spec["density"], spec["hint_zone"])
        flat = [(r, c) for r in range(spec["R"]) for c in range(spec["C"]) if grid[r][c] == 1]
        total = len(flat)
        public = {
            "name": spec["name"], "R": spec["R"], "C": spec["C"], "flammable": grid,
            "budget": spec["budget"], "hint_zone": spec["hint_zone"],
            "wind_bias": list(spec["wind_bias"]), "p_base": spec["p_base"],
            "wind_extra": spec["wind_extra"], "p_wind_dominant": spec["p_wind_dominant"],
            "w_hint": spec["w_hint"], "n_draws": 30,
        }
        out.append({"public": public, "grid": grid, "flat": flat, "total": total,
                     "seed": spec["seed"], "spec": spec})
    return out


# ----------------------------- ignition / wind draws -------------------------
def _draw_scenario(inst, k):
    seed = inst["seed"]
    hz = inst["spec"]["hint_zone"]
    w_hint = inst["spec"]["w_hint"]
    u1 = _rand01(seed, k, 101)
    if u1 < w_hint:
        h = hz["r1"] - hz["r0"] + 1
        w = hz["c1"] - hz["c0"] + 1
        ur = _rand01(seed, k, 102); uc = _rand01(seed, k, 103)
        rr = hz["r0"] + min(int(ur * h), h - 1)
        cc = hz["c0"] + min(int(uc * w), w - 1)
    else:
        flat = inst["flat"]
        ui = _rand01(seed, k, 104)
        idx = min(int(ui * len(flat)), len(flat) - 1)
        rr, cc = flat[idx]
    wind_bias = tuple(inst["spec"]["wind_bias"])
    p_dom = inst["spec"]["p_wind_dominant"]
    u2 = _rand01(seed, k, 105)
    if u2 < p_dom:
        wd = wind_bias
    else:
        others = [d for d in DIRS if d != wind_bias]
        u3 = _rand01(seed, k, 106)
        wd = others[min(int(u3 * 3), 2)]
    return (rr, cc), wd


def _reach_count(inst, firebreak, ignite, wind_dir, draw_idx):
    grid = inst["grid"]; R = inst["public"]["R"]; C = inst["public"]["C"]
    p_base = inst["spec"]["p_base"]; wind_extra = inst["spec"]["wind_extra"]
    seed = inst["seed"]
    r0, c0 = ignite
    if grid[r0][c0] == 0 or (r0, c0) in firebreak:
        return 0
    visited = {(r0, c0)}
    frontier = [(r0, c0)]
    anti = (-wind_dir[0], -wind_dir[1])
    p_down = min(1.0, p_base + wind_extra)
    p_up = max(0.0, p_base - wind_extra)
    while frontier:
        nxt = []
        for (r, c) in frontier:
            for (dr, dc) in DIRS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < R and 0 <= nc < C):
                    continue
                if grid[nr][nc] == 0 or (nr, nc) in firebreak or (nr, nc) in visited:
                    continue
                if (dr, dc) == wind_dir:
                    p = p_down
                elif (dr, dc) == anti:
                    p = p_up
                else:
                    p = p_base
                rv = _rand01(seed, draw_idx, 90000000 + r * 100000 + c, 90000000 + nr * 100000 + nc)
                if rv < p:
                    visited.add((nr, nc))
                    nxt.append((nr, nc))
        frontier = nxt
    return len(visited)


def _mean_burn(inst, firebreak):
    n = inst["public"]["n_draws"]
    total = inst["total"]
    if total == 0:
        return 0.0
    acc = 0.0
    for k in range(n):
        ignite, wd = _draw_scenario(inst, k)
        acc += _reach_count(inst, firebreak, ignite, wd, k)
    return acc / (n * total)


def baseline(inst):
    return _mean_burn(inst, frozenset())


def score(inst, answer):
    if not isinstance(answer, dict):
        return False, None
    cells = answer.get("cells")
    if cells is None:
        cells = []
    if not isinstance(cells, list):
        return False, None
    budget = inst["public"]["budget"]
    if len(cells) > budget:
        return False, None
    R, C = inst["public"]["R"], inst["public"]["C"]
    grid = inst["grid"]
    seen = set()
    fb = set()
    for cell in cells:
        if (not isinstance(cell, list) and not isinstance(cell, tuple)) or len(cell) != 2:
            return False, None
        r, c = cell
        if isinstance(r, bool) or isinstance(c, bool):
            return False, None
        if not isinstance(r, int) or not isinstance(c, int):
            return False, None
        if not (0 <= r < R and 0 <= c < C):
            return False, None
        if grid[r][c] != 1:
            return False, None
        if (r, c) in seen:
            return False, None
        seen.add((r, c))
        fb.add((r, c))
    obj = _mean_burn(inst, frozenset(fb))
    if obj != obj:
        return False, None
    return True, obj


def _candidate_answer(cand, public):
    ans, st = isorun.run_candidate(cand, public, timeout=STEP_TIMEOUT)
    if st != "OK":
        return None
    return ans


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()
    vec = []
    for inst in instances:
        b = baseline(inst)
        ans = _candidate_answer(cand, inst["public"])
        if ans is None:
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False; obj = None
        if not ok:
            vec.append(0.0)
            continue
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        if not (r == r) or r in (float("inf"), float("-inf")) or r < 0:
            vec.append(0.0)
            continue
        vec.append(max(0.0, min(1.0, r)))
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
