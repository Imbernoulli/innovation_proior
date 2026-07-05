import sys, json, math, random, isorun

# ==========================================================================
# fsx_B_0362 -- constrained-OR (Format B, isolated candidate)
# Theme: "ski resort lifts" -- two-echelon spare-parts inventory for the
# critical drive-motor controllers of a large ski resort.  A single central
# warehouse (echelon 0) supplies N chair/gondola lifts, each running a
# continuous-review one-for-one (S-1,S) base-stock policy.  Failures at a
# lift are Poisson; a stock-out means the lift is DOWN until a controller
# arrives (a backorder).  The warehouse pools risk across all lifts.
#
# DECISION: choose the warehouse order-up-to level s0 and each lift's
# order-up-to level s_i (all non-negative integers).
#
# OBJECTIVE (MINIMIZE, dollars/year):
#     C = h0*E[OH_0] + sum_i h_i*E[OH_i]  +  p * sum_i E[BO_i]
#   holding cost of on-hand stock (warehouse + lifts) + downtime penalty
#   on expected backordered controllers at the lifts.
#
# SERVICE CONSTRAINT (hard SLA): sum_i E[BO_i] <= cap  (expected number of
#   lifts-worth of parts short across the resort).  Violation => infeasible => 0.
#
# The two echelons are COUPLED by Sherbrooke's METRIC: warehouse backorders
# inflate each lift's effective resupply lead time (Little's law), so
# starving the warehouse to save central holding cost pushes cost onto the
# lifts.  Everything is computed with EXACT Poisson math -> deterministic.
# ==========================================================================


def pois_ebo_oh(S, m):
    """Exact expected backorders E[(D-S)^+] and expected on-hand E[(S-D)^+]
    for order-up-to S under Poisson(m) lead-time demand D. Returns (ebo, oh).
    Identity: E[OH] - E[BO] = S - m, so oh = ebo + (S - m)."""
    if m <= 0.0:
        return 0.0, float(max(0, S))
    S = int(S)
    # iterate pmf; accumulate tail expectation sum_{x>S}(x-S)p(x)
    xmax = S + int(m + 15.0 * math.sqrt(m) + 80.0)
    logm = math.log(m)
    # start pmf at x=0
    p = math.exp(-m)
    ebo = 0.0
    x = 0
    while x <= xmax:
        if x > S:
            ebo += (x - S) * p
        x += 1
        p *= m / x            # p now = pmf(x)
        if x > S and p < 1e-16 and (x - S) * p < 1e-15:
            # tail contribution negligible
            if p * (x - S) < 1e-18:
                break
    oh = ebo + (S - m)
    if oh < 0.0:
        oh = 0.0
    return ebo, oh


def _evaluate(pub, s0, s_list):
    """Return (feasible, cost, agg_ebo). Full METRIC evaluation."""
    N = pub["N"]
    lam = pub["lam"]; T = pub["T"]; T0 = pub["T0"]
    h = pub["h"]; h0 = pub["h0"]; p = pub["p"]; cap = pub["cap"]
    Lam = pub["Lambda"]
    m0 = Lam * T0
    ebo0, oh0 = pois_ebo_oh(s0, m0)
    delay = ebo0 / Lam if Lam > 0 else 0.0
    holding = h0 * oh0
    shortage_ebo = 0.0
    for i in range(N):
        m_i = lam[i] * (T[i] + delay)
        ebo_i, oh_i = pois_ebo_oh(s_list[i], m_i)
        holding += h[i] * oh_i
        shortage_ebo += ebo_i
    cost = holding + p * shortage_ebo
    feasible = shortage_ebo <= cap * (1.0 + 1e-9)
    return feasible, cost, shortage_ebo


def make_instances():
    N = 30
    out = []
    for si in range(10):
        rng = random.Random(7000 + si)
        lam = [round(rng.uniform(0.3, 3.0), 4) for _ in range(N)]
        T = [round(rng.uniform(0.02, 0.08), 4) for _ in range(N)]
        T0 = round(rng.uniform(0.15, 0.35), 4)
        h = [round(rng.uniform(50.0, 400.0), 2) for _ in range(N)]
        h0 = round(rng.uniform(40.0, 110.0), 2)
        p = round(rng.uniform(150.0, 600.0), 2)
        Lam = round(sum(lam), 6)
        cap = round(rng.uniform(0.40, 1.00), 4)
        pub = {
            "N": N, "lam": lam, "T": T, "T0": T0,
            "h": h, "h0": h0, "p": p, "cap": cap,
            "Lambda": Lam, "S_max": 15, "S0_max": 45,
        }
        out.append({"public": pub, "hidden": {}})
    return out


def _baseline_alloc(pub):
    """Decoupled equal-budget construction (also the 'trivial' strategy):
    stock the warehouse to near-perfect, then give every lift the same tight
    expected-backorder budget cap/N.  Feasible but wasteful."""
    N = pub["N"]; lam = pub["lam"]; T = pub["T"]
    Lam = pub["Lambda"]; T0 = pub["T0"]
    S_max = pub["S_max"]; S0_max = pub["S0_max"]
    m0 = Lam * T0
    s0 = 0
    while s0 < S0_max:
        e0, _ = pois_ebo_oh(s0, m0)
        if e0 <= 0.02:
            break
        s0 += 1
    e0, _ = pois_ebo_oh(s0, m0)
    delay = e0 / Lam if Lam > 0 else 0.0
    budget = pub["cap"] / N * 0.9
    s = []
    for i in range(N):
        m_i = lam[i] * (T[i] + delay)
        si = 0
        while si < S_max:
            e_i, _ = pois_ebo_oh(si, m_i)
            if e_i <= budget:
                break
            si += 1
        s.append(si)
    return s0, s


def baseline(inst):
    pub = inst["public"]
    s0, s = _baseline_alloc(pub)
    _, cost, _ = _evaluate(pub, s0, s)
    return cost


def score(inst, ans):
    pub = inst["public"]
    N = pub["N"]; S_max = pub["S_max"]; S0_max = pub["S0_max"]
    if not isinstance(ans, dict):
        return False, 0.0
    if "s0" not in ans or "s" not in ans:
        return False, 0.0
    s0 = ans["s0"]; s = ans["s"]
    if isinstance(s0, bool) or not isinstance(s0, (int, float)):
        return False, 0.0
    if float(s0) != int(s0):
        return False, 0.0
    s0 = int(s0)
    if s0 < 0 or s0 > S0_max:
        return False, 0.0
    if not isinstance(s, list) or len(s) != N:
        return False, 0.0
    clean = []
    for v in s:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return False, 0.0
        if v != v or v in (float("inf"), float("-inf")):
            return False, 0.0
        if float(v) != int(v):
            return False, 0.0
        v = int(v)
        if v < 0 or v > S_max:
            return False, 0.0
        clean.append(v)
    feasible, cost, _ = _evaluate(pub, s0, clean)
    if not feasible:
        return False, 0.0
    if cost != cost or cost <= 0.0 or cost in (float("inf"), float("-inf")):
        return False, 0.0
    return True, cost


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, stt = isorun.run_candidate(cand, inst["public"], timeout=20)
        if stt != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
