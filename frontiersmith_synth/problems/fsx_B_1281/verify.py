import sys, math

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def eff_tonnes(p):
    price, tonnes, reported, ref_base, irr, threshold, reversal, perm_years, buffer = p
    inflation_penalty = min(1.0, ref_base / reported)
    fin_add = (threshold - irr) / 1000.0
    fin_add = 0.0 if fin_add < 0.0 else (1.0 if fin_add > 1.0 else fin_add)
    additionality = min(inflation_penalty, fin_add)
    survival = (1.0 - reversal / 10000.0) ** perm_years
    permanence = 1.0 - (1.0 - buffer / 100.0) * (1.0 - survival)
    return tonnes * additionality * permanence

def cost_of(p):
    return p[0] * p[1]

def main():
    try:
        itoks = open(sys.argv[1]).read().split()
    except Exception:
        fail("bad input file")
    try:
        it = iter(itoks)
        n = int(next(it))
        budget = int(next(it))
        projects = []
        for _ in range(n):
            row = [int(next(it)) for _ in range(9)]
            projects.append(row)
    except Exception:
        fail("bad input")

    if n <= 0 or budget <= 0:
        fail("degenerate input")
    for p in projects:
        price, tonnes, reported, ref_base, irr, threshold, reversal, perm_years, buffer = p
        if price <= 0 or tonnes <= 0 or reported <= 0 or ref_base <= 0:
            fail("bad input fields")
        if not (0 <= reversal < 10000) or perm_years <= 0 or not (0 <= buffer <= 100):
            fail("bad input fields (range)")

    # ---- internal baseline B: naive "skim the first few listed, buy what fits" ----
    order = range(min(n, 6))
    spent = 0
    triv_set = []
    for idx in order:
        c = cost_of(projects[idx])
        if spent + c <= budget:
            spent += c
            triv_set.append(idx)
    B = sum(eff_tonnes(projects[idx]) for idx in triv_set)
    B = max(1e-9, B)

    # ---- parse participant output ----
    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("bad output file")
    if not otoks:
        fail("empty output")
    try:
        k = int(otoks[0])
    except Exception:
        fail("bad k")
    if k < 0 or k > n:
        fail("k out of range")
    if len(otoks) < 1 + k:
        fail("not enough tokens for k indices")
    idxs = []
    for tok in otoks[1:1 + k]:
        try:
            v = int(tok)
        except Exception:
            fail("non-integer index %r" % tok)
        idxs.append(v)

    seen = set()
    for v in idxs:
        if v < 1 or v > n:
            fail("index out of range %d" % v)
        if v in seen:
            fail("duplicate index %d" % v)
        seen.add(v)

    total_cost = 0
    for v in idxs:
        total_cost += cost_of(projects[v - 1])
    if total_cost > budget:
        fail("over budget: spent %d > budget %d" % (total_cost, budget))

    F = 0.0
    for v in idxs:
        e = eff_tonnes(projects[v - 1])
        if not math.isfinite(e):
            fail("non-finite effective tonnes")
        F += e
    if not math.isfinite(F):
        fail("non-finite objective")

    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("F=%.4f B=%.4f spent=%d/%d Ratio: %.6f" % (F, B, total_cost, budget, sc / 1000.0))

if __name__ == "__main__":
    main()
