#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0535 -- "Set tolls so selfish drivers behave socially"
(family: congestion-marginal-cost-tolling; format B, quality-metric).

THEME.  A city runs a corridor of PARALLEL routes between one origin and one
destination (highway lanes, a bridge, side streets, a tunnel...).  Each route e
has a superlinear travel-time (latency) function

        l_e(f) = a_e * f ** p_e + b_e          (a_e > 0, b_e >= 0, p_e >= 1)

where f is the flow (vehicles) on that route.  Drivers are SELFISH: given a per-
route toll tau_e (a fixed money-time surcharge), every driver picks the route with
the smallest experienced cost l_e(f_e) + tau_e.  The traffic settles into a USER
EQUILIBRIUM (a Wardrop equilibrium): all used routes share one common experienced
cost, no driver can switch and do better.  Total demand splits over the routes.

The city sets ONE toll per route to steer this selfish equilibrium toward the
socially best flow -- the split that MINIMISES the true toll-free total travel
time  sum_e f_e * l_e(f_e).  (Tolls are internal transfers; they do NOT count in
the social cost, they only re-route drivers.)

THE CATCH (scenario-distribution).  Demand is uncertain: it is drawn from a
distribution.  You must commit to ONE toll vector, and it is graded on a held-out
set of demand scenarios you never see.  A toll that is perfect for one demand is
wrong for another, so no fixed toll reaches the per-scenario optimum -- there is
always headroom.

THE INSIGHT (marginal-cost internalisation).  A driver joining route e adds not
only their own delay but slows everyone already there.  The externality they
ignore is  f_e * l_e'(f_e) = a_e * p_e * f_e ** p_e.  The socially optimal toll
equals THIS marginal externality -- proportional to the *marginal* latency,
carrying the exponent p_e -- NOT a level proportional to the observed congestion
a_e * f_e ** p_e (which drops the p_e factor and, on routes with different
exponents, steers flow the wrong way).

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "m": M, "a": [a_0..], "b": [b_0..], "p": [p_0..],
             "train_demands": [D_0, ...]}   # sample demands you MAY use to design tolls
  stdout: ONE JSON object:
            {"tolls": [tau_0, ..., tau_{M-1}]}   # tau_e >= 0, finite
  A toll vector is VALID iff it is a list of exactly M finite numbers, each with
  0 <= tau_e <= 1e9.  Anything else (wrong length, negative, NaN/inf, crash,
  timeout, non-JSON) -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Graded on HIDDEN eval demands.  Per
instance, averaging over the eval demands, we compute:
    L_zero = toll-free total latency at the untolled user equilibrium (do nothing)
    L_so   = toll-free total latency at the per-scenario social optimum (ideal LB)
    L_cand = toll-free total latency at the user equilibrium under YOUR tolls
and normalise (do-nothing -> 0.1, unreachable ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (L_zero - L_cand) / (L_zero - L_so), 0, 1 )
A do-nothing (all-zero) toll scores ~0.1; matching the per-scenario ideal would
score 1.0 but is UNREACHABLE with a single fixed toll over a demand distribution,
so even the marginal-cost toll stays strictly below 1.0.  Tolls that steer flow
worse than doing nothing score below 0.1.

ISOLATION.  The candidate is untrusted and runs OS-sandboxed in a fresh
subprocess via `isorun.run_candidate`; it only ever sees the PUBLIC instance.
The eval demands and all references (L_zero, L_so) are computed by THIS parent
process, so an introspecting/frame-walking candidate learns nothing useful.

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

    def nxt():
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return (state >> 11) / float(1 << 53)      # uniform [0,1)

    return nxt


def _u(nxt, lo, hi):
    return lo + (hi - lo) * nxt()


# ----------------------------- instance family -----------------------------
# Each spec: (seed, m, p_pattern, center, spread, trap)
#   p_pattern -> per-link exponents; "hetero" mixes flat (p=1) and steep (p=4)
#   links so the marginal factor p_e differs sharply across routes (the trap).
_SPECS = [
    (1101, 5, [1.0, 4.0, 1.0, 4.0, 2.0],           30.0, 0.35, True),   # trap
    (2202, 6, [1.0, 3.0, 1.0, 3.0, 1.0, 3.0],      45.0, 0.40, True),   # trap
    (3303, 4, [1.0, 4.0, 2.0, 4.0],                24.0, 0.30, True),   # trap
    (4404, 5, [3.0, 3.0, 3.0, 3.0, 3.0],           36.0, 0.30, False),
    (5505, 6, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],      50.0, 0.35, False),
    (6606, 4, [2.0, 3.0, 2.0, 3.0],                20.0, 0.28, False),
    (7707, 7, [1.0, 4.0, 1.0, 4.0, 2.0, 3.0, 1.0], 60.0, 0.40, True),   # trap
    (8808, 5, [4.0, 4.0, 4.0, 4.0, 4.0],           30.0, 0.30, False),
    (9909, 6, [1.0, 2.0, 4.0, 1.0, 2.0, 4.0],      40.0, 0.45, True),   # trap
    (1011, 5, [2.0, 2.0, 3.0, 3.0, 1.0],           28.0, 0.30, False),
]


def _build_instance(seed, m, ppat, center, spread, trap):
    nxt = _rng(seed)
    p = list(ppat)
    ref = center / m                 # nominal per-route flow
    a = []
    b = []
    for e in range(m):
        coeff = _u(nxt, 2.0, 6.0)    # congestion term ~ coeff at the nominal flow
        a.append(coeff * ref ** (-p[e]))
        b.append(_u(nxt, 1.0, 5.0))  # free-flow latency
    lo = center * (1.0 - spread)
    hi = center * (1.0 + spread)
    train = [_u(nxt, lo, hi) for _ in range(6)]      # public sample scenarios
    ev = [_u(nxt, lo, hi) for _ in range(8)]         # HIDDEN graded scenarios
    return {"name": f"corridor{seed}", "m": m, "a": a, "b": b, "p": p,
            "train_demands": train, "eval_demands": ev}


def _build_instances():
    return [_build_instance(*s) for s in _SPECS]


# ----------------------------- equilibrium solvers -------------------------
def _solve(a, beff, p, D, iters=200):
    """Common-cost split: cost_e(f) = a_e f^p_e + beff_e; return flows summing to
    D with all used routes at a common cost level (monotone bisection)."""
    m = len(a)
    if D <= 0.0:
        return [0.0] * m
    lo = min(beff)
    hi = min(beff[e] + a[e] * D ** p[e] for e in range(m))
    if hi <= lo:
        hi = lo + 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = 0.0
        for e in range(m):
            if mid > beff[e]:
                s += ((mid - beff[e]) / a[e]) ** (1.0 / p[e])
        if s < D:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    f = []
    for e in range(m):
        f.append(((lam - beff[e]) / a[e]) ** (1.0 / p[e]) if lam > beff[e] else 0.0)
    return f


def _ue(a, b, p, tolls, D):
    beff = [b[e] + tolls[e] for e in range(len(b))]
    return _solve(a, beff, p, D)


def _so(a, b, p, D):
    # marginal social cost m_e(f) = a_e (p_e+1) f^p_e + b_e -> same solver form
    ap = [a[e] * (p[e] + 1.0) for e in range(len(a))]
    return _solve(ap, b, p, D)


def _total_latency(a, b, p, f):
    tot = 0.0
    for e in range(len(f)):
        tot += f[e] * (a[e] * f[e] ** p[e] + b[e])
    return tot


# ----------------------------- validation ----------------------------------
def _valid_tolls(inst, answer):
    if not isinstance(answer, dict):
        return None
    t = answer.get("tolls")
    if not isinstance(t, list) or len(t) != inst["m"]:
        return None
    out = []
    for x in t:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        x = float(x)
        if x != x or x in (float("inf"), float("-inf")):
            return None
        if x < 0.0 or x > 1e9:
            return None
        out.append(x)
    return out


# ----------------------------- scoring driver ------------------------------
def _score_instance(inst, tolls):
    a, b, p = inst["a"], inst["b"], inst["p"]
    ev = inst["eval_demands"]
    Lz = Ls = Lc = 0.0
    for D in ev:
        Lz += _total_latency(a, b, p, _ue(a, b, p, [0.0] * inst["m"], D))
        Ls += _total_latency(a, b, p, _so(a, b, p, D))
        Lc += _total_latency(a, b, p, _ue(a, b, p, tolls, D))
    n = len(ev)
    Lz, Ls, Lc = Lz / n, Ls / n, Lc / n
    denom = Lz - Ls
    if denom < 1e-9:
        denom = 1e-9
    r = 0.1 + 0.9 * (Lz - Lc) / denom
    if r != r or r in (float("inf"), float("-inf")):
        return 0.0
    return max(0.0, min(1.0, r))


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    vec = []
    for inst in _build_instances():
        public = {"name": inst["name"], "m": inst["m"],
                  "a": list(inst["a"]), "b": list(inst["b"]),
                  "p": list(inst["p"]), "train_demands": list(inst["train_demands"])}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            tolls = _valid_tolls(inst, ans)
        except Exception:
            tolls = None
        if tolls is None:
            vec.append(0.0)
            continue
        try:
            r = _score_instance(inst, tolls)
        except Exception:
            r = 0.0
        vec.append(r)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
