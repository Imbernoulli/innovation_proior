#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0852 -- "Sleeper Solvers: Budgeting a Node-Expansion
Portfolio" (family: portfolio-expansion-budget; format B, quality-metric).

THEME.  One hard planning instance, a PORTFOLIO of heuristic solvers.  Each solver
h has a hidden, deterministic, non-decreasing QUALITY CURVE curve_h(k) = the best
solution quality it reaches if run ALONE from scratch for k total node-expansions.
The curve's *shape* differs per solver per instance and is unknown; you only get a
short INFORMATIVENESS PROBE -- curve_h evaluated at a handful of small checkpoint
budgets, well below the total budget B.  You must decide, once, how to split the
REST of B across the portfolio (BUDGET-ALLOCATION-ACROSS-SOLVERS).  Only the final
total each solver receives matters (curves are pure functions of cumulative
expansions), so PROGRESS-TRIGGERED SWITCHING is realized as a pre-committed
allocation informed by each probe's trend: a heuristic whose probe is small but
ACCELERATING is a possible late bloomer worth funding over one that is already
flattening out despite a louder current value (HEURISTIC-INFORMATIVENESS-PROBE).

CURVE FAMILIES (hidden; only samples are exposed via probes).
  "sat" (saturating):  Qmax * (1 - exp(-k / tau))              -- rises then caps.
  "lin" (linear-capped): min(Qmax, a * k)                       -- steady, no cap yet.
  "sig" (delayed sigmoid): a logistic with hidden threshold k0 and width s, shifted
        so curve(0) = 0 exactly -- looks near-zero until k approaches k0, then rises
        steeply toward Qmax.  With k0 set as a fraction of B, its probe window shows
        only a faint, ACCELERATING trace: exactly the "sleeper" signal a probe-driven
        strategy must catch and a magnitude-only strategy misses.

Roles per instance (H=6 solvers): one "decoy" (fast sat, low Qmax -- the visible
trap), three "dud" (fast sat, tiny Qmax), one "steady" (slow lin filler, never the
true winner), and one TRUE WINNER which is either "leader" (a sat curve with tau
scaled to B, so it is both visibly ahead in the probe window AND actually best --
an "easy", non-trap instance) or "delayed" (a sig curve with k0 scaled to B, so the
decoy looks best throughout the probe window while delayed is truly the strongest
at full budget -- a TRAP instance).  Solver identity is shuffled per instance so
position carries no information.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : {"name": str, "budget": B (int>0),
           "heuristics": [{"id": str, "checkpoints": [k0..k3] (increasing ints,
                            all << B), "probe": [curve_h(k0), ..., curve_h(k3)]}, ...]}
  stdout: {"alloc": {id: expansions (int >= 0), ...}}   # missing id -> 0
          Every id must be one of the instance's solver ids, every value a
          non-boolean finite non-negative int, and sum(alloc.values()) <= B.
          Any violation, crash, timeout, or non-JSON output -> that instance
          scores 0.0.

SCORING (deterministic; no wall-time).  Per instance, with the TRUE curves (only
ever touched in this parent process):
    quality = max_h curve_h( alloc.get(h.id, 0) )
    best    = max_h curve_h(B)     -- best single solver in hindsight (oracle ceiling)
    worst   = min_h curve_h(B)     -- worst single solver in hindsight (weak floor)
    r = clamp( 0.1 + 0.9 * (quality - worst) / max(best - worst, 1e-9), 0, 1 )

ISOLATION.  Candidates run in a FRESH SANDBOXED SUBPROCESS via
isorun.run_candidate; they only ever see the PUBLIC instance (budget, ids,
checkpoints, probes).  The true curve parameters, `best`, and `worst` never leave
this parent process, so a frame-walking / introspecting candidate learns nothing
that helps it allocate.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


def _unit(ni):
    return ni(1, 1000000) / 1000000.0


# ----------------------------- curve families -------------------------------
def _f_sat(k, Qmax, tau):
    return Qmax * (1.0 - math.exp(-k / tau))


def _f_lin(k, Qmax, a):
    return min(Qmax, a * k)


def _f_sig(k, Qmax, k0, s):
    base = 1.0 / (1.0 + math.exp(k0 / s))
    val = (1.0 / (1.0 + math.exp(-(k - k0) / s)) - base) / max(1e-9, (1.0 - base))
    return max(0.0, Qmax * val)


def _eval_curve(curve, k):
    kind, p1, p2, p3 = curve
    if kind == "sat":
        return _f_sat(k, p1, p2)
    if kind == "lin":
        return _f_lin(k, p1, p2)
    if kind == "sig":
        return _f_sig(k, p1, p2, p3)
    raise ValueError(kind)


def _build_heuristic(role, ni, B):
    if role == "decoy":              # fast, capped, mediocre -- the visible trap
        Qmax = 0.30 + 0.16 * _unit(ni)
        tau = 6 + ni(0, 8)
        return ("sat", Qmax, tau, None)
    if role == "delayed":            # invisible in probe window, high ceiling -- sleeper
        Qmax = 0.80 + 0.17 * _unit(ni)
        k0 = B * (0.16 + 0.10 * _unit(ni))
        s = B * (0.030 + 0.020 * _unit(ni))
        return ("sig", Qmax, k0, s)
    if role == "leader":             # visibly rising AND genuinely best -- honest winner
        Qmax = 0.80 + 0.17 * _unit(ni)
        tau = B * (0.075 + 0.045 * _unit(ni))
        return ("sat", Qmax, tau, None)
    if role == "steady":             # slow linear filler, never the true winner
        Qmax = 0.42 + 0.16 * _unit(ni)
        a = Qmax / (B * (1.6 + 0.9 * _unit(ni)))
        return ("lin", Qmax, a, None)
    if role == "dud":                # always poor
        Qmax = 0.04 + 0.06 * _unit(ni)
        tau = 2 + ni(0, 4)
        return ("sat", Qmax, tau, None)
    raise ValueError(role)


def _checkpoints_for(B):
    fracs = (0.010, 0.025, 0.050, 0.090)
    cs = sorted(set(max(2, int(round(B * f))) for f in fracs))
    while len(cs) < 4:
        cs.append(cs[-1] + 1)
    return cs[:4]


# ----------------------------- instance family -----------------------------
# (seed, roles for 6 solvers, budget B). >=6 of 10 are TRAP instances (true
# winner is "delayed" but "decoy" dominates the whole probe window); the rest
# are "easy" instances where the true winner ("leader") is also visibly ahead,
# including a couple of larger held-out budgets for generalization.
_SPECS = [
    (301, ["decoy", "dud", "dud", "dud", "steady", "delayed"], 2600),   # trap
    (302, ["steady", "decoy", "dud", "dud", "delayed", "dud"], 3200),   # trap
    (303, ["dud", "steady", "delayed", "decoy", "dud", "dud"], 2200),   # trap
    (304, ["decoy", "steady", "delayed", "dud", "dud", "dud"], 3600),   # trap, larger B
    (305, ["steady", "delayed", "decoy", "dud", "dud", "dud"], 2900),   # trap
    (306, ["dud", "dud", "decoy", "delayed", "dud", "steady"], 3400),   # trap
    (401, ["leader", "dud", "decoy", "dud", "steady", "dud"], 1400),    # easy
    (402, ["decoy", "dud", "dud", "leader", "dud", "steady"], 1600),    # easy
    (403, ["dud", "leader", "dud", "steady", "dud", "decoy"], 1200),    # easy
    (502, ["steady", "decoy", "delayed", "dud", "dud", "dud"], 4000),   # trap, held-out largest
]


def _build_instance(seed, roles, B):
    ni = _rng(seed)
    curves = [_build_heuristic(r, ni, B) for r in roles]
    idx = list(range(len(roles)))
    for i in range(len(idx) - 1, 0, -1):        # deterministic Fisher-Yates shuffle
        j = ni(0, i)
        idx[i], idx[j] = idx[j], idx[i]
    curves = [curves[i] for i in idx]
    ids = [f"h{i}" for i in range(len(roles))]
    cks = _checkpoints_for(B)
    probes = [[round(_eval_curve(c, k), 6) for k in cks] for c in curves]
    return {"name": f"portfolio{seed}", "budget": B, "ids": ids,
            "checkpoints": cks, "curves": curves, "probes": probes}


def _build_instances():
    return [_build_instance(seed, roles, B) for (seed, roles, B) in _SPECS]


# ----------------------------- validation & scoring -------------------------
def _validate(inst, answer):
    """Return the alloc dict {id: int} if valid, else None."""
    if not isinstance(answer, dict):
        return None
    alloc = answer.get("alloc")
    if not isinstance(alloc, dict):
        return None
    ids = set(inst["ids"])
    if not set(alloc.keys()) <= ids:
        return None
    out = {}
    total = 0
    for hid in inst["ids"]:
        v = alloc.get(hid, 0)
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        if v < 0:
            return None
        out[hid] = v
        total += v
    if total > inst["budget"]:
        return None
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        B = inst["budget"]
        curves = inst["curves"]
        ids = inst["ids"]
        # NOTE: no "name"/instance-id field is exposed -- the candidate must reason
        # from budget + probes, not memorize a per-instance lookup key.
        public = {
            "budget": B,
            "heuristics": [
                {"id": ids[i], "checkpoints": list(inst["checkpoints"]),
                 "probe": list(inst["probes"][i])}
                for i in range(len(ids))
            ],
        }
        ans, st = isorun.run_candidate(cand, public, timeout=5)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            alloc = _validate(inst, ans)
        except Exception:
            alloc = None
        if alloc is None:
            vec.append(0.0)
            continue
        quality = max(_eval_curve(c, alloc[ids[i]]) for i, c in enumerate(curves))
        best = max(_eval_curve(c, B) for c in curves)
        worst = min(_eval_curve(c, B) for c in curves)
        r = 0.1 + 0.9 * (quality - worst) / max(best - worst, 1e-9)
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        vec.append(max(0.0, min(1.0, r)))

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
