#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0360 -- "Railway Freight Yard: Budget-Constrained
Motive-Power Routing".

Family: offline-decision-policy (Frontier-CS cant_be_late/cloudcast/llm_router/
llm_sql anchor -- here the cost-aware ROUTING flavour), skinned as a hump-yard
dispatch desk. A fixed daily TRACE of freight "cuts" (blocks of cars) must each be
pulled from the receiving yard over the hump and delivered. For every cut the desk
may assign ONE of several motive-power options (a yard switcher, a road-slug set, a
mainline unit, ...). Assigning cut i to option j burns an integer amount of diesel
`fuel[i][j]` and delivers integer throughput value `value[i][j]` (ton-miles moved
before the cutoff). Options are listed in INCREASING fuel order; heavier power moves
more tonnage but with DIMINISHING returns, and how steeply value grows with fuel
varies from cut to cut. The desk has a fixed daily diesel BUDGET B.

The desk must choose, offline over the whole trace, ONE option per cut so that the
TOTAL delivered value is maximised while TOTAL fuel <= B. This is a multiple-choice
knapsack: assigning every cut its cheapest (option 0) switcher is always affordable
but leaves most tonnage on the ground; a value/fuel-density greedy that upgrades each
cut straight to its single best option ignores the diminishing-returns structure and
wastes budget on inefficient heavy units; a Lagrangian / DP-style policy that weighs
intermediate options across the whole trace does markedly better. There is no easy
optimum and several viable strategies -- exactly the offline sequential-decision /
closed-form cost-minimisation shape that generalises to unseen traces.

The candidate is UNTRUSTED: it is run as an ISOLATED stdin->stdout subprocess via
`isorun`, so it only ever sees the public instance and can never reach the
evaluator's frames / instance generator / scorer. All scoring is deterministic
(seeded instance generation, pure integer arithmetic; no wall-time).

Scoring (maximisation; higher delivered value is better):
  obj(assign) = total delivered value of a feasible assignment (sum fuel <= B)
  baseline b  = obj(all-cheapest = every cut on option 0)   (computed by evaluator)
  For a FEASIBLE assignment with objective obj:  r = min(1, 0.1 * obj / b)
  -> the all-cheapest policy maps to exactly 0.1; a policy delivering k times the
     cheapest-policy value maps to min(1, 0.1*k). An infeasible (over-budget) or
     malformed answer scores 0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun

MASK = (1 << 64) - 1


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & MASK

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & MASK
        return lo + (state >> 17) % (hi - lo + 1)
    return nxt


# ----------------------------- instance family -----------------------------
def make_instances():
    """Deterministic, seeded. Returns list of {'public':..., 'hidden':{}}.

    For each cut we draw M motive-power options in INCREASING fuel order (a base
    switcher plus successively heavier units). The delivered value starts at a small
    base and grows with each heavier option by a DIMINISHING marginal amount
    (marginal ~ gain/k with per-cut `gain` and small noise), so value is concave in
    fuel and the cheapest upgrade steps are the most fuel-efficient. The per-cut
    `gain` varies widely, so which cuts are worth upgrading -- and to WHICH option --
    is a genuine cross-trace allocation problem, not a per-cut local choice.

    The daily budget B is set to the all-cheapest fuel plus 35% of the head-room to
    the all-heaviest fuel: enough to upgrade a fraction of the trace but never all of
    it. The all-cheapest policy is therefore always feasible (baseline floor) and the
    all-heaviest policy is always over budget (used by the `invalid` tier)."""
    specs = [
        # (seed, N cuts, M options) -- last four are larger held-out traces
        (3101, 12, 3), (3102, 14, 4), (3103, 13, 4), (3104, 15, 4),
        (3105, 12, 5), (3106, 16, 4), (3107, 17, 4), (3108, 16, 5),
        (3109, 18, 4), (3110, 18, 5), (3111, 20, 4), (3112, 20, 5),
    ]
    alpha_num, alpha_den = 35, 100
    out = []
    for seed, N, M in specs:
        r = _rng(seed)
        fuel = []
        value = []
        for _ in range(N):
            f0 = r(5, 10)
            fs = [f0]
            for _k in range(1, M):
                fs.append(fs[-1] + r(4, 11))          # strictly increasing fuel
            v0 = r(3, 9)
            gain = r(16, 44)                           # per-cut value scale (varies a lot)
            vals = [v0]
            cur = v0
            for k in range(1, M):
                marg = int(round(gain / k)) + r(-2, 4)  # diminishing marginal value
                marg = max(1, marg)
                cur = cur + marg
                vals.append(cur)
            fuel.append(fs)
            value.append(vals)
        minf = sum(fuel[i][0] for i in range(N))
        maxf = sum(fuel[i][-1] for i in range(N))
        B = minf + (alpha_num * (maxf - minf)) // alpha_den
        public = {"n_cuts": N, "n_options": M, "budget": B,
                  "fuel": fuel, "value": value}
        out.append({"public": public, "hidden": {}})
    return out


# ----------------------------- scoring -------------------------------------
def _total_value(p, assign):
    return sum(p["value"][i][assign[i]] for i in range(p["n_cuts"]))


def _total_fuel(p, assign):
    return sum(p["fuel"][i][assign[i]] for i in range(p["n_cuts"]))


def baseline(inst):
    """Objective of the trivial all-cheapest policy (every cut on option 0)."""
    p = inst["public"]
    assign = [0] * p["n_cuts"]
    return _total_value(p, assign)


def score(inst, answer):
    """Strictly validate the answer against the instance; return (ok, obj)."""
    p = inst["public"]
    N = p["n_cuts"]
    M = p["n_options"]
    if not isinstance(answer, dict):
        return False, None
    assign = answer.get("assign", None)
    if not isinstance(assign, list) or len(assign) != N:
        return False, None
    norm = []
    for x in assign:
        if isinstance(x, bool) or not isinstance(x, int):
            return False, None
        if x < 0 or x >= M:
            return False, None
        norm.append(x)
    if _total_fuel(p, norm) > p["budget"]:
        return False, None                          # over budget -> infeasible
    obj = _total_value(p, norm)
    if obj != obj or obj < 0:
        return False, None
    return True, obj


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, None
        if not ok or obj is None:
            vec.append(0.0)
            continue
        b = baseline(inst)
        if b <= 0:
            vec.append(0.0)
            continue
        r = min(1.0, 0.1 * obj / b)
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
