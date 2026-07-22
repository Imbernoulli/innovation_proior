# TIER: strong
import sys, json, random, math

def cell_cost(eff_table, target, lam, types, R, C, r, c, t):
    cost = abs(eff_table[t] - target[r][c])
    if c + 1 < C:
        d = t - types[r][c + 1]; cost += lam * d * d
    if c - 1 >= 0:
        d = t - types[r][c - 1]; cost += lam * d * d
    if r + 1 < R:
        d = t - types[r + 1][c]; cost += lam * d * d
    if r - 1 >= 0:
        d = t - types[r - 1][c]; cost += lam * d * d
    return cost

def main():
    inst = json.load(sys.stdin)
    R, C, K = inst["R"], inst["C"], inst["K"]
    eff_table = inst["eff_table"]
    target = inst["target"]
    lam = inst["interface_weight"]

    # start from independent nearest-match, then jointly re-optimize with a
    # smoothness-aware annealed local search (poor-man's MRF / graph-cut):
    # every move is scored against BOTH the cell's own target AND its
    # current neighbors' chosen types, so the search can trade a little
    # pointwise accuracy for a much cheaper interface bill.
    types = [[0] * C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            tv = target[r][c]
            types[r][c] = min(range(K), key=lambda t: abs(eff_table[t] - tv))

    rng = random.Random(1234567 + R * 97 + C)
    cells = [(r, c) for r in range(R) for c in range(C)]
    iters = 300
    T0, T1 = 0.2, 0.002
    for it in range(iters):
        temp = T0 * ((T1 / T0) ** (it / max(1, iters - 1)))
        rng.shuffle(cells)
        for (r, c) in cells:
            cur = types[r][c]
            cur_cost = cell_cost(eff_table, target, lam, types, R, C, r, c, cur)
            if rng.random() < 0.7:
                cand_t = max(0, min(K - 1, cur + rng.choice([-1, 1])))
            else:
                cand_t = rng.randrange(K)
            new_cost = cell_cost(eff_table, target, lam, types, R, C, r, c, cand_t)
            if new_cost <= cur_cost or rng.random() < math.exp(-(new_cost - cur_cost) / max(temp, 1e-9)):
                types[r][c] = cand_t

    # final deterministic ICM polish pass (no randomness, monotone improvement)
    for _ in range(5):
        changed = False
        for r in range(R):
            for c in range(C):
                best_t = types[r][c]
                best_c = cell_cost(eff_table, target, lam, types, R, C, r, c, best_t)
                for t in range(K):
                    cc_ = cell_cost(eff_table, target, lam, types, R, C, r, c, t)
                    if cc_ < best_c - 1e-12:
                        best_c = cc_; best_t = t
                if best_t != types[r][c]:
                    types[r][c] = best_t; changed = True
        if not changed:
            break

    print(json.dumps({"types": types}))

main()
