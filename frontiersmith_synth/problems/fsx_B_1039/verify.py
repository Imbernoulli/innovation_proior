#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic scorer for the market-garden
rotation-plan problem. Prints '... Ratio: <float>' on the LAST line and exits 0.
"""
import sys, json, math


def die0(reason):
    print("INFEASIBLE: %s Ratio: 0.0" % reason)
    sys.exit(0)


def load_instance(path):
    with open(path) as f:
        return json.load(f)


def is_finite_number(x):
    """True for finite ints/floats, WITHOUT ever converting an arbitrary-precision
    JSON int through float() -- math.isfinite(10**400) raises OverflowError, which
    would otherwise crash the checker instead of yielding the promised 0.0 score."""
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return -10 ** 18 <= x <= 10 ** 18
    if isinstance(x, float):
        return math.isfinite(x)
    return False


def parse_plan(raw, P, T, C):
    try:
        obj = json.loads(raw)
    except Exception:
        return None, "output is not valid JSON"
    if not isinstance(obj, dict) or "plan" not in obj:
        return None, "missing 'plan' key"
    plan = obj["plan"]
    if not isinstance(plan, list) or len(plan) != P:
        return None, "plan must have exactly P rows"
    out = []
    for row in plan:
        if not isinstance(row, list) or len(row) != T:
            return None, "each plot row must have exactly T entries"
        r = []
        for v in row:
            if not is_finite_number(v):
                return None, "non-finite / non-numeric crop index"
            iv = int(v)
            if float(iv) != float(v):
                return None, "crop index must be an integer"
            if iv < 0 or iv >= C:
                return None, "crop index out of range"
            r.append(iv)
        out.append(r)
    return out, None


def yield_norm(soil, req, K):
    r = 1.0
    for k in range(K):
        if req[k] > 1e-9:
            r = min(r, soil[k] / req[k])
    return max(0.0, min(1.0, r))


def simulate(inst, plan):
    """Replay the exact public dynamics; returns total market-weighted revenue."""
    P, T, K = inst["P"], inst["T"], inst["K"]
    cap = inst["cap"]
    crops = inst["crops"]
    pest_grow, pest_decay = inst["pest_grow"], inst["pest_decay"]
    pest_cap, pest_coeff = inst["pest_cap"], inst["pest_coeff"]
    threshold = inst["glut_threshold"]

    soil = [list(row) for row in inst["init_soil"]]
    pest_p = [0.0] * P
    prev_family = [-1] * P

    total = 0.0
    for t in range(T):
        # phase 1: everyone's crop choice + physical yield_norm + pest multiplier
        # (pest_p here is the value carried over from the END of season t-1)
        choices = []
        for p in range(P):
            c = crops[plan[p][t]]
            yn = yield_norm(soil[p], c["req"], K)
            pmult = max(0.0, 1.0 - pest_coeff * pest_p[p])
            y_phys = c["base_yield"] * yn * pmult
            choices.append((c, yn, y_phys))

        # phase 2: cross-plot diversification / market-glut pricing, per family,
        # computed simultaneously across all plots this season
        fam_count = {}
        for (c, yn, y_phys) in choices:
            fam_count[c["family"]] = fam_count.get(c["family"], 0) + 1
        for (c, yn, y_phys) in choices:
            cnt = fam_count[c["family"]]
            glut_mult = min(1.0, threshold / cnt) if cnt > 0 else 1.0
            total += c["price"] * y_phys * glut_mult

        # phase 3: physical soil depletion/replenishment (glut only affects price,
        # not the physical harvest draw) + pest-memory update
        for p in range(P):
            c, yn, y_phys = choices[p]
            for k in range(K):
                soil[p][k] = max(0.0, min(cap[k], soil[p][k] - c["depletion"][k] * yn + c["replenish"][k]))
            if c["family"] == prev_family[p]:
                pest_p[p] = min(pest_cap, pest_p[p] + pest_grow)
            else:
                pest_p[p] = max(0.0, pest_p[p] - pest_decay)
            prev_family[p] = c["family"]
    return total


def baseline_plan(inst):
    """Checker's own trivial construction: monoculture of crop 0 on every plot,
    every season -- ignores soil, pest and market dynamics entirely."""
    P, T = inst["P"], inst["T"]
    return [[0] * T for _ in range(P)]


def main():
    if len(sys.argv) < 3:
        print("usage: verify.py <in> <out> <ans> Ratio: 0.0")
        sys.exit(0)
    in_path, out_path = sys.argv[1], sys.argv[2]
    inst = load_instance(in_path)
    P, T, K = inst["P"], inst["T"], inst["K"]
    C = len(inst["crops"])

    try:
        with open(out_path) as f:
            raw = f.read()
    except Exception:
        die0("cannot read output file")

    plan, err = parse_plan(raw, P, T, C)
    if err:
        die0(err)

    F = simulate(inst, plan)
    if not math.isfinite(F) or F < 0:
        die0("non-finite or negative objective")

    B = simulate(inst, baseline_plan(inst))
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    ratio = sc / 1000.0
    print("garden=%s F=%.4f B=%.4f Ratio: %.6f" % (inst.get("name", "?"), F, B, ratio))
    sys.exit(0)


if __name__ == "__main__":
    main()
