import sys, json, math, random, isorun

# ==========================================================================
# fsx_B_0864 -- solver-portfolio-allocator (Format B, isolated candidate)
# Theme: "Spend a compute budget across rival solvers"
#
# Each instance is one portfolio SESSION: a shared step budget `budget` and a
# batch of `n_cases` public cases (cheap features: domain in {0,1}, size in
# [0,1]).  There are k=4 deterministic sub-solvers.  Solver j needs
#     req_j(case) = max(floor, base_j + size_coef_j*size + domain_coef_j*domain
#                              + inter_coef_j*size*domain + noise_j[case])
# steps to finish `case` (all coefficients + noise are PUBLIC, given in the
# instance).  The candidate submits an ORDERED schedule of attempts
# (case, solver, amount); the evaluator plays it back against the shared
# budget in order, deducting `amount` from what remains and crediting it to
# that (case, solver)'s running spend.  A case is SOLVED the moment some
# solver's cumulative credited spend on it reaches that solver's requirement.
# Objective: maximize the fraction of cases solved.  Order matters because
# once the shared budget is exhausted, later attempts in the schedule never
# run -- so a schedule that front-loads doomed/expensive attempts starves
# the cheap wins behind it.
#
# The planted trap: one "niche" solver is mediocre ON AVERAGE but, on a
# domain x size INTERACTION region, becomes far cheaper than every other
# solver (including the solver with the best average cost).  Always backing
# the single globally-best-average solver (the "obvious" policy) -- or
# spreading the budget thin and even -- both miss this, and get badly beaten
# by a policy that computes each case's OWN cheapest solver and schedules by
# ascending cost.
# ==========================================================================

K = 4


def make_instances():
    # (n_cases, budget, is_trap, p_domain1)
    specs = [
        (30, 420, False, 0.50),
        (30, 420, False, 0.50),
        (30, 420, True,  0.65),
        (30, 420, True,  0.70),
        (30, 430, False, 0.50),
        (34, 470, True,  0.65),
        (28, 400, False, 0.45),
        (32, 460, True,  0.60),
        (30, 430, False, 0.50),
        (30, 410, True,  0.65),
    ]
    floor = 9.0
    out = []
    for si, (C, T, trap, pdom) in enumerate(specs):
        rng = random.Random(9100 + si)
        profiles = []
        for j in range(K):
            if j == 0:                                    # the "niche" solver
                base = round(40.0 + rng.uniform(-2, 2), 4)
                sizec = round(26.0 + rng.uniform(-2, 2), 4)
                domc = round(-2.0 + rng.uniform(-2, 2), 4)
                interc = round((-52.0 if trap else -6.0) + rng.uniform(-3, 3), 4)
            elif j == 2:                                  # the "champion" (best on average)
                base = round(18.0 + rng.uniform(-2, 2), 4)
                sizec = round(10.0 + rng.uniform(-2, 2), 4)
                domc = round((3.0 if trap else 0.0) + rng.uniform(-2, 2), 4)
                interc = round(0.0 + rng.uniform(-2, 2), 4)
            else:                                         # filler solvers
                base = round(26.0 + rng.uniform(-3, 5), 4)
                sizec = round(14.0 + rng.uniform(-4, 8), 4)
                domc = round(rng.uniform(-6, 6), 4)
                interc = round(rng.uniform(-6, 6), 4)
            profiles.append({"base": base, "size_coef": sizec,
                              "domain_coef": domc, "inter_coef": interc})
        cases = []
        noise = []
        for c in range(C):
            dom = 1 if rng.random() < pdom else 0
            if trap and dom == 1:
                size = rng.uniform(0.6, 1.0)
            elif trap and dom == 0:
                size = rng.uniform(0.0, 0.4)
            else:
                size = rng.random()
            cases.append({"domain": dom, "size": round(size, 5)})
            noise.append([round(rng.uniform(-2, 2), 4) for _ in range(K)])
        pub = {"k": K, "n_cases": C, "budget": T, "req_floor": floor,
               "cases": cases, "solver_profiles": profiles, "case_noise": noise}
        out.append({"public": pub, "hidden": {}})
    return out


def _req(pub, ci, j):
    p = pub["solver_profiles"][j]
    c = pub["cases"][ci]
    v = (p["base"] + p["size_coef"] * c["size"] + p["domain_coef"] * c["domain"]
         + p["inter_coef"] * c["size"] * c["domain"] + pub["case_noise"][ci][j])
    return max(pub["req_floor"], v)


def _play(pub, attempts):
    """Replay an ordered attempt list against the shared budget. Returns
    (n_solved, C). Already-solved cases silently ignore further attempts
    (no charge)."""
    C = pub["n_cases"]
    remaining = float(pub["budget"])
    spent = [[0.0] * pub["k"] for _ in range(C)]
    solved = [False] * C
    for ci, j, amt in attempts:
        if solved[ci] or amt <= 0.0:
            continue
        if amt > remaining + 1e-9:
            continue
        remaining -= amt
        spent[ci][j] += amt
        if spent[ci][j] + 1e-9 >= _req(pub, ci, j):
            solved[ci] = True
        if remaining <= 1e-9:
            break
    return sum(solved), C


def baseline(inst):
    """Naive 'don't think about it' recipe: always fund the SAME fixed
    default solver (index 1), one full per-case share, in case order."""
    pub = inst["public"]
    C, T = pub["n_cases"], pub["budget"]
    share = T / C
    attempts = [(ci, 1, share) for ci in range(C)]
    n, C2 = _play(pub, attempts)
    return n / C2


def _near_optimal(inst):
    """Reference upper anchor: per-case argmin solver, scheduled by
    ascending cost (provably optimal for this shared-budget / fixed-cost /
    unit-value model -- an exchange argument). Used ONLY to calibrate score
    headroom; a candidate can at best tie it."""
    pub = inst["public"]
    C = pub["n_cases"]
    costs = []
    for ci in range(C):
        bj, bc = 0, None
        for j in range(pub["k"]):
            r = _req(pub, ci, j)
            if bc is None or r < bc:
                bc, bj = r, j
        costs.append((bc, ci, bj))
    costs.sort(key=lambda t: (t[0], t[1]))
    attempts = [(ci, j, c) for (c, ci, j) in costs]
    n, C2 = _play(pub, attempts)
    return n / C2


def score(inst, ans):
    pub = inst["public"]
    C, k = pub["n_cases"], pub["k"]
    if not isinstance(ans, dict) or "attempts" not in ans:
        return False, 0.0
    raw = ans["attempts"]
    if not isinstance(raw, list) or len(raw) > 20000:
        return False, 0.0
    attempts = []
    for a in raw:
        if not isinstance(a, list) or len(a) != 3:
            return False, 0.0
        ci_r, j_r, amt = a
        # bool is a subclass of int in Python but is NOT an accepted numeric
        # index/amount type here -- reject explicitly.
        if isinstance(ci_r, bool) or isinstance(j_r, bool) or isinstance(amt, bool):
            return False, 0.0
        if not isinstance(ci_r, (int, float)) or not isinstance(j_r, (int, float)):
            return False, 0.0
        if not isinstance(amt, (int, float)):
            return False, 0.0
        ci_f, j_f, amt_f = float(ci_r), float(j_r), float(amt)
        for v in (ci_f, j_f, amt_f):
            if v != v or v in (float("inf"), float("-inf")):
                return False, 0.0
        # bounds-check the RAW value first (with a tight tolerance) so a
        # value that only rounds into range (e.g. -0.0000005 rounding to 0,
        # or C-0.0000005 rounding to C) is rejected as out-of-range rather
        # than silently snapped to a valid index.
        if ci_f < -1e-6 or ci_f > (C - 1) + 1e-6:
            return False, 0.0
        if j_f < -1e-6 or j_f > (k - 1) + 1e-6:
            return False, 0.0
        if abs(ci_f - round(ci_f)) > 1e-6 or abs(j_f - round(j_f)) > 1e-6:
            return False, 0.0
        ci, j = int(round(ci_f)), int(round(j_f))
        ci = min(max(ci, 0), C - 1)
        j = min(max(j, 0), k - 1)
        if amt_f < 0.0:
            return False, 0.0
        attempts.append((ci, j, amt_f))
    n_solved, _ = _play(pub, attempts)
    obj = n_solved / C
    if obj != obj:
        return False, 0.0
    return True, obj


def main():
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
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        no = _near_optimal(inst)
        ceil = min(0.97, no * 1.15)
        denom = max(ceil - b, 0.05)
        r = 0.1 + 0.75 * (obj - b) / denom
        r = max(0.0, min(1.0, r))
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
