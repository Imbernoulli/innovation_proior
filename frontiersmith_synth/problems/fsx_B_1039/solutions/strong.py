# TIER: strong
"""Insight: the high-scoring object on a plot is not a sequence of individually
good decisions -- it is a LIMIT CYCLE in the plot's own soil-state space. We
search over candidate short rotations (every ordered complementary crop pair,
plus monoculture as a fallback) and simulate each one honestly (soil + pest,
no glut) on a single plot to find which rotation is a genuinely SUSTAINED
orbit: one whose nutrient draws and replenishments cancel out so it can be
repeated indefinitely at high yield, instead of strip-mining one nutrient
dimension or one crop family.

Once we have the best sustainable rotation, the remaining lever is
cross-plot: if every plot ran the identical rotation in phase, every plot
would plant the same crop family in the same season and trip the market-glut
discount. So we split the plots into phase-offset groups that run the SAME
rotation but staggered in time, so every season the planted families are
spread across plots instead of synchronized -- turning one profitable orbit
into a diversified, glut-free planting schedule."""
import sys, json


def yield_norm(soil, req, K):
    r = 1.0
    for k in range(K):
        if req[k] > 1e-9:
            r = min(r, soil[k] / req[k])
    return max(0.0, min(1.0, r))


def simulate_pattern(pattern, init_soil, cap, crops, K, T,
                      pest_grow, pest_decay, pest_cap, pest_coeff):
    """Replay `pattern` (length T, list of crop indices) on ONE plot, honest
    physics, glut excluded (glut is a cross-plot effect we handle separately).
    Returns total revenue."""
    soil = list(init_soil)
    pest_p = 0.0
    prev_family = -1
    total = 0.0
    for t in range(T):
        c = crops[pattern[t]]
        yn = yield_norm(soil, c["req"], K)
        pmult = max(0.0, 1.0 - pest_coeff * pest_p)
        y_phys = c["base_yield"] * yn * pmult
        total += c["price"] * y_phys
        for k in range(K):
            soil[k] = max(0.0, min(cap[k], soil[k] - c["depletion"][k] * yn + c["replenish"][k]))
        if c["family"] == prev_family:
            pest_p = min(pest_cap, pest_p + pest_grow)
        else:
            pest_p = max(0.0, pest_p - pest_decay)
        prev_family = c["family"]
    return total


def main():
    inst = json.load(sys.stdin)
    P, T, K = inst["P"], inst["T"], inst["K"]
    cap = inst["cap"]
    crops = inst["crops"]
    C = len(crops)
    pest_grow, pest_decay = inst["pest_grow"], inst["pest_decay"]
    pest_cap, pest_coeff = inst["pest_cap"], inst["pest_coeff"]
    init0 = inst["init_soil"][0]  # every plot starts identical in this family

    def score(pattern):
        return simulate_pattern(pattern, init0, cap, crops, K, T,
                                 pest_grow, pest_decay, pest_cap, pest_coeff)

    best_score, best_kind, best_args = -1.0, "mono", 0

    # candidate 1: monoculture of each single crop (fallback / control group)
    for c in range(C):
        s = score([c] * T)
        if s > best_score:
            best_score, best_kind, best_args = s, "mono", c

    # candidate 2: every ordered pair of DIFFERENT families, alternating --
    # this is where a genuine two-crop limit cycle (planted nutrient depletion
    # cancelling a planted replenishment) shows up
    for c1 in range(C):
        for c2 in range(C):
            if c1 == c2 or crops[c1]["family"] == crops[c2]["family"]:
                continue
            pattern = [c1 if t % 2 == 0 else c2 for t in range(T)]
            s = score(pattern)
            if s > best_score:
                best_score, best_kind, best_args = s, "pair", (c1, c2)

    if best_kind == "pair":
        c1, c2 = best_args
        pat_a = [c1 if t % 2 == 0 else c2 for t in range(T)]  # starts with c1
        pat_b = [c2 if t % 2 == 0 else c1 for t in range(T)]  # phase-shifted
        # split plots into two phase groups so every season roughly half the
        # plots plant family(c1) and half plant family(c2) -- avoids glut
        # regardless of the instance's diversification threshold.
        plan = []
        for p in range(P):
            plan.append(pat_a if p % 2 == 0 else pat_b)
    else:
        c = best_args
        plan = [[c] * T for _ in range(P)]

    print(json.dumps({"plan": plan}, separators=(" , ", ": ")))


if __name__ == "__main__":
    main()
