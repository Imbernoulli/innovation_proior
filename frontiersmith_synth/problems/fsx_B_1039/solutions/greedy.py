# TIER: greedy
"""Per-plot myopic 1-step planner: every season, on every plot INDEPENDENTLY,
plant whichever crop maximizes THIS season's price*yield given the plot's own
current soil and pest state. No lookahead, no coordination across plots.
This is the "obvious first approach" an average strong coder writes -- and
because every plot starts from the identical soil state, all plots compute
the identical argmax every season and move in lockstep: they all pile onto
the single highest-value crop, tripping the market-glut penalty every season,
they repeat the same crop family back to back (pest spiral), and they run the
soil's limiting nutrient toward zero with nothing to replenish it."""
import sys, json


def yield_norm(soil, req, K):
    r = 1.0
    for k in range(K):
        if req[k] > 1e-9:
            r = min(r, soil[k] / req[k])
    return max(0.0, min(1.0, r))


def main():
    inst = json.load(sys.stdin)
    P, T, K = inst["P"], inst["T"], inst["K"]
    cap = inst["cap"]
    crops = inst["crops"]
    pest_grow, pest_decay = inst["pest_grow"], inst["pest_decay"]
    pest_cap, pest_coeff = inst["pest_cap"], inst["pest_coeff"]

    soil = [list(row) for row in inst["init_soil"]]
    pest_p = [0.0] * P
    prev_family = [-1] * P
    plan = [[0] * T for _ in range(P)]

    for t in range(T):
        for p in range(P):
            best_ci, best_v = 0, -1.0
            for ci, c in enumerate(crops):
                yn = yield_norm(soil[p], c["req"], K)
                pmult = max(0.0, 1.0 - pest_coeff * pest_p[p])
                v = c["price"] * c["base_yield"] * yn * pmult
                if v > best_v + 1e-12:
                    best_v, best_ci = v, ci
            plan[p][t] = best_ci
            c = crops[best_ci]
            yn = yield_norm(soil[p], c["req"], K)
            for k in range(K):
                soil[p][k] = max(0.0, min(cap[k], soil[p][k] - c["depletion"][k] * yn + c["replenish"][k]))
            if c["family"] == prev_family[p]:
                pest_p[p] = min(pest_cap, pest_p[p] + pest_grow)
            else:
                pest_p[p] = max(0.0, pest_p[p] - pest_decay)
            prev_family[p] = c["family"]

    print(json.dumps({"plan": plan}, separators=(" , ", ": ")))


if __name__ == "__main__":
    main()
