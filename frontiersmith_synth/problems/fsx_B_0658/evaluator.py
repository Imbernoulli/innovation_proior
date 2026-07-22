import sys, json, random, statistics, isorun

# ==========================================================================
# fsx_B_0658 -- variance-adaptive-quadrature-budget (Format B, isolated)
# Theme: allocate a fixed extra sampling budget across domain regions whose
# hidden per-sample noise variance is wildly heterogeneous, to minimize the
# expected squared error of a stratified estimate of a total integral.
#
# Each region r has a public width w_r and a HIDDEN true noise std sigma_r.
# The candidate is given a small PILOT batch of real noisy measurements
# (drawn i.i.d. from Normal(mu_r, sigma_r^2), mu_r also hidden) per region,
# plus a total extra-sample budget B, and must output how to split B across
# regions. If it later took alloc[r] more i.i.d. measurements in region r
# and averaged all (pilot + extra) samples, the resulting stratified
# estimator of the total integral I = sum_r w_r*mu_r has (unbiased, exact)
# variance
#     Var = sum_r  w_r^2 * sigma_r^2 / (pilot_size + alloc[r])
# which is what the grader scores (objective: MINIMIZE this variance). The
# grader computes this analytically from the hidden sigma_r -- it does not
# need to draw the extra samples, so scoring is exact and deterministic.
# ==========================================================================

PILOT_SIZE = 5


def make_instances():
    # (num_regions, budget_multiplier, has_variance_trap)
    specs = [
        (6, 8, False),
        (7, 8, True),
        (8, 7, True),
        (6, 9, False),
        (9, 7, True),
        (7, 8, True),
        (8, 8, False),
        (10, 6, True),
        (6, 9, True),
        (9, 7, False),
    ]
    out = []
    for si, (R, bmult, trap) in enumerate(specs):
        rng = random.Random(20000 + si * 97)
        trap_idx = set()
        if trap:
            k = 1 if R <= 7 else 2
            trap_idx = set(rng.sample(range(R), k))
        widths, sigmas, mus = [], [], []
        for r in range(R):
            if r in trap_idx:
                # small region, but a huge hidden noise spike: uniform /
                # width-proportional allocation both starve it, wasting the
                # budget on the calm, wide regions.
                w = round(rng.uniform(0.2, 0.5), 3)
                sigma = round(rng.uniform(35.0, 60.0), 3)
            else:
                w = round(rng.uniform(1.2, 4.5), 3)
                sigma = round(rng.uniform(0.5, 2.2), 3)
            mu = round(rng.uniform(-5.0, 5.0), 3)
            widths.append(w)
            sigmas.append(sigma)
            mus.append(mu)
        B = int(bmult * R)
        pilots = []
        for r in range(R):
            pilots.append([round(rng.gauss(mus[r], sigmas[r]), 4) for _ in range(PILOT_SIZE)])
        pub = {
            "regions": [{"width": widths[r], "pilot": pilots[r]} for r in range(R)],
            "budget": B,
            "pilot_size": PILOT_SIZE,
        }
        hidden = {"sigmas": sigmas, "mus": mus}
        out.append({"public": pub, "hidden": hidden})
    return out


def _variance(pub, sigmas, alloc):
    p = pub["pilot_size"]
    total = 0.0
    for reg, sigma, a in zip(pub["regions"], sigmas, alloc):
        w = reg["width"]
        total += (w * w) * (sigma * sigma) / (p + a)
    return total


def baseline(inst):
    pub = inst["public"]
    R = len(pub["regions"])
    return _variance(pub, inst["hidden"]["sigmas"], [0.0] * R)


def score(inst, ans):
    pub = inst["public"]
    R = len(pub["regions"])
    B = pub["budget"]
    if not isinstance(ans, dict) or "alloc" not in ans:
        return False, 0.0
    alloc = ans["alloc"]
    if not isinstance(alloc, list) or len(alloc) != R:
        return False, 0.0
    clean = []
    for v in alloc:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return False, 0.0
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            return False, 0.0
        if v < -1e-9:
            return False, 0.0
        clean.append(max(v, 0.0))
    if sum(clean) > B * (1.0 + 1e-6) + 1e-6:
        return False, 0.0
    var = _variance(pub, inst["hidden"]["sigmas"], clean)
    if var != var or var <= 0.0:
        return False, 0.0
    return True, var


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
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
