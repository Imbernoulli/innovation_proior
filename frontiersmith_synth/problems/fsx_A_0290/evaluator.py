#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0290 -- "Ridgeline Watch: Budgeted Fire-Tower Placement"
(family: heuristic-contest-offline; format B, quality-metric; theme: forest fire watchtowers).

THEME.  A forestry service must place a network of fire-lookout towers over a
mountain valley.  The valley is a grid of forest cells; each cell has an integer
FIRE-RISK weight (dry, wind-exposed ridgelines are high, damp shaded gullies are
low).  There is a fixed menu of candidate tower SITES; a tower built at a site
watches every cell inside a square patrol range (its line-of-sight radius) and a
bigger, taller tower (larger radius) costs more to raise and staff.  The service
has a fixed capital BUDGET.  Building a subset of towers whose total cost stays
within budget, it wants to MAXIMIZE the total fire-risk weight of the forest cells
that are watched by at least one built tower (a cell watched by several towers still
counts once -- redundant overlap is wasted money).

This is *budgeted maximum coverage* skinned as fire-tower siting: sites = sets (the
cells each tower watches), weights = cell values, tower cost = set cost, and we
maximize the weight of the covered union subject to a knapsack budget.  It is
NP-hard; a coverage-blind rule is easily beaten, plain ratio-greedy leaves value on
the table where overlap and lumpy costs interact, and local search over
add/drop/swap moves does better still -- yet the all-towers coverage ceiling stays
out of reach under any real budget, so scores keep headroom below 1.0.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "N": int, "M": int, "B": int,
             "weight": [[w_00, ...], ...],   # N x N grid, w >= 0
             "tx": [x_0, ...],  "ty": [y_0, ...],   # site column / row, 0..N-1
             "tr": [r_0, ...],  "tc": [c_0, ...]}   # patrol radius / build cost, per site
          Tower j watches every cell (row, col) with |row - ty[j]| <= tr[j] and
          |col - tx[j]| <= tr[j] (clipped to the grid): a (2r+1)x(2r+1) square.
  stdout: ONE JSON object:
            {"build": [j0, j1, ...]}
          the list of DISTINCT site indices to build.  Order is irrelevant.

  A plan is VALID iff `build` is a list of distinct integers in [0, M) whose total
  cost sum(tc[j]) <= B.  Duplicate indices, an out-of-range index, over budget, a
  crash, a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  For each instance we compute two references
in THIS parent process:
    q_base  = watched weight of the internal COST-ASCENDING FILL operator
              (buy the cheapest sites first while the budget lasts -- coverage-blind)
    q_full  = watched weight if EVERY site were built (the union ceiling, ignoring
              budget; generally unaffordable, hence unreachable)
  Let q_cand be the watched weight of the candidate's valid plan.  We normalize with
  an affine anchor (weak fill -> 0.1, union ceiling -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(1e-9, q_full - q_base), 0, 1 )
  Matching the cost-ascending fill scores ~0.1; reaching the (budget-infeasible)
  all-towers ceiling scores 1.0; doing worse than the fill scores < 0.1.

ISOLATION.  The candidate is untrusted and runs in a FRESH SANDBOXED SUBPROCESS via
`isorun.run_candidate`, seeing ONLY the public instance.  The references (q_base,
q_full) are computed by this parent, so a frame-walking / source-reading candidate
learns nothing that helps it inflate its score.

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

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


# ----------------------------- instance family -----------------------------
def _build_weights(nxt, N, n_hot):
    """Low ambient fire-risk plus a few decaying hotspot ridges."""
    W = [[nxt(0, 2) for _ in range(N)] for _ in range(N)]
    for _ in range(n_hot):
        cy = nxt(0, N - 1); cx = nxt(0, N - 1)
        peak = nxt(12, 30); rad = nxt(3, 6)
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                r = cy + dr; c = cx + dc
                if 0 <= r < N and 0 <= c < N:
                    d = abs(dr) + abs(dc)
                    add = peak - (peak * d) // (rad + 1)
                    if add > 0:
                        W[r][c] += add
    return W


def _build_towers(nxt, N, M):
    """M candidate sites: random cell, patrol radius 2..5, cost grows with area."""
    tx, ty, tr, tc = [], [], [], []
    for _ in range(M):
        tx.append(nxt(0, N - 1))
        ty.append(nxt(0, N - 1))
        r = nxt(2, 5)
        tr.append(r)
        # cost ~ area of the patrol square plus lumpy noise, so cost/coverage
        # trade-offs are nontrivial (big towers watch more but cost more)
        tc.append(4 + r * r + nxt(0, 8))
    return tx, ty, tr, tc


def _build_instances():
    """Deterministic instance family: (seed, N, M, n_hot, budget_frac_pct)."""
    specs = [
        (2901, 24, 34,  5, 16),
        (2902, 26, 38,  6, 15),
        (2903, 28, 40,  6, 14),
        (2904, 24, 34,  4, 18),
        (2905, 30, 44,  7, 14),
        (2906, 26, 38,  5, 17),
        (2907, 28, 42,  6, 15),
        (2908, 30, 46,  8, 13),
        # harder / larger held-out instances (more sites, tighter budgets)
        (2951, 34, 56,  9, 12),
        (2952, 32, 52,  8, 12),
        (2953, 36, 60, 10, 11),
        (2954, 34, 58,  9, 13),
    ]
    out = []
    for seed, N, M, n_hot, bpct in specs:
        nxt = _rng(seed)
        W = _build_weights(nxt, N, n_hot)
        tx, ty, tr, tc = _build_towers(nxt, N, M)
        B = (sum(tc) * bpct) // 100          # budget affords only a fraction of sites
        out.append({"name": f"valley{seed}", "N": N, "M": M, "B": B,
                    "weight": W, "tx": tx, "ty": ty, "tr": tr, "tc": tc})
    return out


# ----------------------------- coverage helpers ----------------------------
def _rect(inst, j):
    N = inst["N"]; r = inst["tr"][j]
    r0 = max(0, inst["ty"][j] - r); r1 = min(N - 1, inst["ty"][j] + r)
    c0 = max(0, inst["tx"][j] - r); c1 = min(N - 1, inst["tx"][j] + r)
    return r0, r1, c0, c1


def _coverage_weight(inst, built):
    """Total weight of cells watched by at least one built site."""
    N = inst["N"]; W = inst["weight"]
    seen = [[False] * N for _ in range(N)]
    total = 0
    for j in built:
        r0, r1, c0, c1 = _rect(inst, j)
        row = seen
        for r in range(r0, r1 + 1):
            sr = row[r]; wr = W[r]
            for c in range(c0, c1 + 1):
                if not sr[c]:
                    sr[c] = True
                    total += wr[c]
    return total


# ----------------------------- references ----------------------------------
def _cost_ascending_fill(inst):
    """Weak coverage-blind operator: buy cheapest sites first until budget is gone."""
    tc = inst["tc"]; B = inst["B"]
    order = sorted(range(inst["M"]), key=lambda j: (tc[j], j))
    built = []; spent = 0
    for j in order:
        if spent + tc[j] <= B:
            built.append(j); spent += tc[j]
    return _coverage_weight(inst, built)


def _full_ceiling(inst):
    return _coverage_weight(inst, list(range(inst["M"])))


# ----------------------------- validation ----------------------------------
def _valid_plan_weight(inst, answer):
    """Validate the candidate answer; return watched weight or None if infeasible."""
    if not isinstance(answer, dict):
        return None
    build = answer.get("build")
    if not isinstance(build, list):
        return None
    M = inst["M"]; tc = inst["tc"]
    seen = set(); spent = 0
    for j in build:
        if isinstance(j, bool) or not isinstance(j, int):
            return None
        if j < 0 or j >= M:
            return None
        if j in seen:
            return None
        seen.add(j)
        spent += tc[j]
    if spent > inst["B"]:
        return None
    return _coverage_weight(inst, list(seen))


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        q_base = _cost_ascending_fill(inst)
        q_full = _full_ceiling(inst)
        denom = q_full - q_base
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "N": inst["N"], "M": inst["M"], "B": inst["B"],
                  "weight": [list(row) for row in inst["weight"]],
                  "tx": list(inst["tx"]), "ty": list(inst["ty"]),
                  "tr": list(inst["tr"]), "tc": list(inst["tc"])}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _valid_plan_weight(inst, ans)
        except Exception:
            q_cand = None
        if q_cand is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (q_cand - q_base) / denom
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
