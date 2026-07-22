#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_N_0809 -- "Anticipatory Tolls: Steering the Commuter Equilibrium"
(family: equilibrium-steering-toll; format B, quality-metric).

THEME.  A city has E parallel commute routes (edges) between one origin and one
destination.  N commuters travel every round.  Each edge e has a CONVEX latency
function l_e(x) = a_e*x^2 + b_e*x + c_e (a_e,b_e,c_e >= 0): the more flow x on the
edge, the slower it gets, and it gets slower faster as load grows (congestion).

The controller (candidate) does NOT get to react in real time -- it must commit,
in one shot, to a full T-round toll schedule BEFORE the episode runs. Every round,
the commuter population re-splits itself among the edges via a fixed, deterministic
best-response rule (SELFISH FLOW BEST-RESPONSE): commuters compare last round's
realized latency plus this round's posted toll, and MORE of them re-route onto
whichever edge currently looks cheap, in proportion to how cheap it looks. A
naive controller that reacts to whichever edge was congested LAST round (tolling
the current hotspot) is always one step behind this feedback loop: taxing today's
jam shoves the crowd onto a parallel edge that becomes tomorrow's jam, and total
realized travel time (which does NOT count the toll -- tolls are a steering price,
not a real cost) suffers. The insight that wins is to work out the SYSTEM-OPTIMAL
split up front (the split that minimizes total travel time, found by equalizing
each edge's MARGINAL cost -- the classic congestion-pricing argument) and price
the crowd's own best-response into landing there directly, instead of dampening
whatever is hot right now.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
    {"name": str, "E": int, "N": float, "T": int, "rho": float,
     "edges": [{"a":.., "b":.., "c":..}, ...]   # length E, all >= 0
     "x0": [float, ...],                        # length E, sums to N
     "toll_max": float}
  stdout: ONE JSON object:
    {"tolls": [[t_0e0, t_0e1, ...], [t_1e0, ...], ...]}   # T lists of length E

  `tolls[t][e]` is the toll posted on edge e for round t (0-indexed, t=0..T-1),
  and must satisfy 0 <= tolls[t][e] <= toll_max and be finite. Any malformed
  output (wrong shape, non-finite, out of range), a crash, a timeout, or
  non-JSON output makes that instance score 0.0.

DYNAMICS (deterministic; computed by the evaluator from the candidate's toll
schedule -- this is exactly what the statement describes, so a candidate can
self-simulate it to plan ahead):
  x^0 = x0 (given).  For round t = 0..T-1:
    cost_e = l_e(x_e^t) + tolls[t][e]                        (perceived cost)
    share_e = (1/cost_e) / sum_j(1/cost_j)                   (best-response split)
    x_e^{t+1} = (1-rho)*x_e^t + rho*N*share_e
    realized_t = sum_e x_e^{t+1} * l_e(x_e^{t+1})             (TRUE travel time; toll excluded)
  objective = sum_t realized_t   (MINIMIZE)

SCORING (deterministic; no wall-time). Per instance:
    B = objective of the ALL-ZERO toll schedule (evaluator's own internal baseline)
    r = clamp( 0.2 * B / candidate_objective, 0, 1 )
  A do-nothing candidate scores ~0.2. Beating the baseline (B/obj > 1) scores
  above 0.2; there is headroom above any of our reference solutions (checked at
  authoring time), so scores do not saturate at 1.0.

ISOLATION. Candidates run OS-sandboxed via isorun.run_candidate; they only ever
see the PUBLIC instance. All ten instances + the baseline computation live only
in this (parent) process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun

SCALE = 0.2          # normalizes a do-nothing (zero-toll) candidate to ~0.2


# ----------------------------- latency + dynamics ---------------------------
def l_edge(a, b, c, x):
    return a * x * x + b * x + c


def simulate(E, T, N, rho, edges, x0, tolls):
    """Run the T-round best-response dynamic under a given toll schedule.
    Returns total realized travel time (toll excluded)."""
    x = list(x0)
    total = 0.0
    for t in range(T):
        toll_t = tolls[t]
        cost = [l_edge(edges[e][0], edges[e][1], edges[e][2], x[e]) + toll_t[e] for e in range(E)]
        inv = [1.0 / max(v, 1e-12) for v in cost]
        s = sum(inv)
        share = [v / s for v in inv]
        x = [(1.0 - rho) * x[e] + rho * N * share[e] for e in range(E)]
        total += sum(x[e] * l_edge(edges[e][0], edges[e][1], edges[e][2], x[e]) for e in range(E))
    return total


def zero_tolls(E, T):
    return [[0.0] * E for _ in range(T)]


# ----------------------------- system-optimal reference --------------------
def so_flow(E, N, edges, iters=200):
    """Water-filling bisection: find the flow split that minimizes total travel
    time sum_e x_e*l_e(x_e), by equalizing marginal cost MC_e(x_e)=3a_e x^2+2b_e x+c_e
    across all edges carrying positive flow (the classic convex system-optimum
    condition)."""
    def x_of_mu(mu, e):
        a, b, c = edges[e]
        A = 3.0 * a
        B = 2.0 * b
        C = c - mu
        if C >= 0:
            return 0.0
        if A <= 0:
            return 0.0
        disc = B * B - 4 * A * C
        if disc < 0:
            return 0.0
        r = (-B + math.sqrt(disc)) / (2 * A)
        return max(0.0, r)

    lo, hi = 0.0, 1.0
    while sum(x_of_mu(hi, e) for e in range(E)) < N and hi < 1e18:
        hi *= 2.0
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if sum(x_of_mu(mid, e) for e in range(E)) < N:
            lo = mid
        else:
            hi = mid
    mu = (lo + hi) / 2.0
    xs = [x_of_mu(mu, e) for e in range(E)]
    ssum = sum(xs)
    if ssum > 1e-9:
        xs = [v * N / ssum for v in xs]
    return xs


# ----------------------------- instance family ------------------------------
def _build_instances():
    specs = []
    # BENIGN (7 of 10): moderate reconsideration rate, edges with a WIDE spread of
    # congestion-steepness, all demand starting on the edge the system optimum wants
    # LEAST of. A naive "equalize current travel times" toll recipe helps noticeably
    # here (it does push flow the right general direction) but does not find the
    # true cost-minimizing split, which is far from a cost-equalized split when
    # edges are this heterogeneous.
    specs.append(dict(name="b1", E=2, N=700.0, T=18, rho=0.25,
                       edges=[(0.004, 0.05, 1.0), (1.0, 0.05, 1.0)], x0=[700.0, 0.0]))
    specs.append(dict(name="b2", E=2, N=900.0, T=18, rho=0.22,
                       edges=[(0.004, 0.05, 1.1), (1.2, 0.05, 1.1)], x0=[900.0, 0.0]))
    specs.append(dict(name="b3", E=3, N=800.0, T=18, rho=0.25,
                       edges=[(0.003, 0.04, 0.8), (0.6, 0.06, 1.3), (1.2, 0.08, 1.8)],
                       x0=[800.0, 0.0, 0.0]))
    specs.append(dict(name="b4", E=3, N=1100.0, T=18, rho=0.25,
                       edges=[(0.0025, 0.04, 0.8), (0.7, 0.06, 1.2), (1.4, 0.08, 1.8)],
                       x0=[1100.0, 0.0, 0.0]))
    specs.append(dict(name="b5", E=2, N=550.0, T=18, rho=0.28,
                       edges=[(0.005, 0.05, 0.9), (1.5, 0.05, 0.9)], x0=[550.0, 0.0]))
    specs.append(dict(name="b6", E=4, N=1300.0, T=18, rho=0.25,
                       edges=[(0.003, 0.03, 0.7), (0.4, 0.05, 1.1), (0.9, 0.07, 1.5), (1.6, 0.09, 2.0)],
                       x0=[1300.0, 0.0, 0.0, 0.0]))
    # held-out / larger benign instance
    specs.append(dict(name="h1", E=5, N=2200.0, T=20, rho=0.22,
                       edges=[(0.0025, 0.03, 0.5), (0.3, 0.05, 1.0), (0.7, 0.07, 1.5),
                              (1.2, 0.09, 2.0), (2.0, 0.11, 2.5)],
                       x0=[2200.0, 0.0, 0.0, 0.0, 0.0]))
    # TRAP (3 of 10): higher reconsideration rate. Whichever edge looks cheapest
    # right now pulls a LARGE share of the crowd next round; a controller that
    # reactively taxes today's most-congested edge just pushes that overshoot onto
    # a parallel edge, which then becomes next round's jam -- so reactive damping
    # roughly breaks even with doing nothing, while pricing in the system-optimal
    # split up front settles the flow fast and cuts total travel time sharply.
    specs.append(dict(name="t1", E=2, N=650.0, T=14, rho=0.60,
                       edges=[(0.005, 0.05, 1.0), (0.4, 0.05, 1.0)], x0=[650.0, 0.0]))
    specs.append(dict(name="t2", E=2, N=850.0, T=14, rho=0.55,
                       edges=[(0.006, 0.06, 1.2), (0.48, 0.06, 1.2)], x0=[850.0, 0.0]))
    # held-out / larger trap instance
    specs.append(dict(name="t3", E=3, N=1500.0, T=16, rho=0.60,
                       edges=[(0.004, 0.04, 0.8), (0.32, 0.05, 1.5), (0.6, 0.06, 2.0)],
                       x0=[1500.0, 0.0, 0.0]))
    return specs


# ----------------------------- validation ----------------------------------
def _validate_answer(answer, E, T, toll_max):
    if not isinstance(answer, dict):
        return None
    tolls = answer.get("tolls")
    if not isinstance(tolls, list) or len(tolls) != T:
        return None
    out = []
    for row in tolls:
        if not isinstance(row, list) or len(row) != E:
            return None
        vals = []
        for v in row:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return None
            fv = float(v)
            if fv != fv or fv in (float("inf"), float("-inf")):
                return None
            if fv < -1e-9 or fv > toll_max + 1e-6:
                return None
            vals.append(max(0.0, fv))
        out.append(vals)
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        E, T, N, rho = inst["E"], inst["T"], inst["N"], inst["rho"]
        edges = inst["edges"]
        x0 = inst["x0"]
        # generous toll cap: far above anything a sensible strategy needs
        worst_edge_cost = max(l_edge(a, b, c, N) for (a, b, c) in edges)
        toll_max = 50.0 * worst_edge_cost + 1000.0

        B = simulate(E, T, N, rho, edges, x0, zero_tolls(E, T))

        public = {"name": inst["name"], "E": E, "N": N, "T": T, "rho": rho,
                  "edges": [{"a": a, "b": b, "c": c} for (a, b, c) in edges],
                  "x0": list(x0), "toll_max": toll_max}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            tolls = _validate_answer(ans, E, T, toll_max)
        except Exception:
            tolls = None
        if tolls is None:
            vec.append(0.0)
            continue
        try:
            obj = simulate(E, T, N, rho, edges, x0, tolls)
        except Exception:
            vec.append(0.0)
            continue
        if not (obj == obj) or obj in (float("inf"), float("-inf")) or obj < 0:
            vec.append(0.0)
            continue
        r = SCALE * B / max(obj, 1e-9)
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        vec.append(max(0.0, min(1.0, r)))

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
