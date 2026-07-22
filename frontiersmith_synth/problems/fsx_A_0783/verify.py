#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for fsx_A_0783
(outbreak-prepositioned-stockpile). ans is an unused placeholder.

Instance format (stdin, produced by gen.py):
  N K T ALPHA_PERCENT BUDGET
  P_1 .. P_N                      (one population per line)
  M
  u v w   (M lines; undirected edge, 0-indexed, per-mille contact weight)
  seed_city BETA_PERCENT          (K lines, one per scenario)

Submission (stdout): exactly N whitespace-separated non-negative integers d_1..d_N,
the number of doses prepositioned in city i (sum(d_i) <= BUDGET).

Objective (minimize): worst-case total infections F = max over scenarios of the
population that is EVER infected, under a discrete-time metapopulation SIR run with
dose-based protection p_i = min(1, d_i / cost_i), cost_i = ceil(P_i * ALPHA_PERCENT/100).
Protection multiplies BOTH i's susceptibility and i's infectivity by (1-p_i), so a
fully-dosed city (p_i=1) is a true graph separator: disease cannot pass through it in
either direction.

Baseline B = the SAME simulation with zero doses everywhere (the checker's own
"do nothing" construction). Score: sc = min(1000, 100*B/max(1e-9,F)); Ratio = sc/1000.
"""
import sys, math

BETA_LOCAL = 0.45
GAMMA = 0.18
MAX_PROTECTION = 0.50   # doses reduce but never fully eliminate transmission risk;
                         # kept below 1 - GAMMA/BETA_LOCAL so a single fully-dosed city
                         # can never locally self-extinguish its own outbreak -- only
                         # network-level separator choices contain a cascade

def fail(msg):
    print(f"INVALID: {msg}")
    print("Ratio: 0.0")
    sys.exit(0)

def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    def nx():
        return next(it)
    N = int(nx()); K = int(nx()); T = int(nx())
    alpha_percent = int(nx()); budget = int(nx())
    pops = [int(nx()) for _ in range(N)]
    M = int(nx())
    edges = []
    for _ in range(M):
        u = int(nx()); v = int(nx()); w = int(nx())
        edges.append((u, v, w))
    scenarios = []
    for _ in range(K):
        s = int(nx()); bpct = int(nx())
        scenarios.append((s, bpct))
    return N, K, T, alpha_percent, budget, pops, edges, scenarios

def ceil_cost(pop, alpha_percent):
    return -(-pop * alpha_percent // 100)

def build_adj(N, edges):
    adj = [[] for _ in range(N)]
    for (u, v, w) in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj

def simulate_total_infected(N, pops, adj, seed_city, beta_scale, p, T):
    S = [1.0] * N
    I = [0.0] * N
    inv_pop_seed = 1.0 / pops[seed_city]
    S[seed_city] = 1.0 - inv_pop_seed
    I[seed_city] = inv_pop_seed
    for _t in range(T):
        new_inf = [0.0] * N
        for i in range(N):
            pi = p[i]
            local = BETA_LOCAL * (1.0 - pi) * I[i]
            cross = 0.0
            for (j, w) in adj[i]:
                cross += (w / 1000.0) * beta_scale * (1.0 - pi) * (1.0 - p[j]) * I[j]
            foi = local + cross
            if foi < 0.0:
                foi = 0.0
            ni = foi if foi < S[i] else S[i]
            new_inf[i] = ni
        for i in range(N):
            rec = GAMMA * I[i]
            S[i] -= new_inf[i]
            I[i] = I[i] + new_inf[i] - rec
            R_unused = rec
        # numerical guard: clamp tiny negative drift from floating point
        for i in range(N):
            if S[i] < 0.0:
                S[i] = 0.0
    total = 0.0
    for i in range(N):
        total += pops[i] * (1.0 - S[i])
    return total

def worst_case(N, pops, adj, scenarios, p, T):
    best = 0.0
    for (seed_city, bpct) in scenarios:
        v = simulate_total_infected(N, pops, adj, seed_city, bpct / 100.0, p, T)
        if v > best:
            best = v
    return best

def main():
    if len(sys.argv) < 3:
        fail("usage: verify.py <in> <out> <ans>")
    in_path, out_path = sys.argv[1], sys.argv[2]
    N, K, T, alpha_percent, budget, pops, edges, scenarios = read_instance(in_path)
    adj = build_adj(N, edges)
    costs = [ceil_cost(pops[i], alpha_percent) for i in range(N)]

    raw = open(out_path).read().split()
    if len(raw) != N:
        fail(f"expected exactly {N} integers, got {len(raw)}")
    doses = []
    for tok in raw:
        try:
            d = int(tok)
        except ValueError:
            fail(f"non-integer token: {tok!r}")
        if d < 0:
            fail(f"negative dose count: {d}")
        doses.append(d)
    total_doses = sum(doses)
    if total_doses > budget:
        fail(f"budget exceeded: sum(doses)={total_doses} > budget={budget}")

    p = [min(MAX_PROTECTION, doses[i] / costs[i]) for i in range(N)]
    F = worst_case(N, pops, adj, scenarios, p, T)

    p0 = [0.0] * N
    B = worst_case(N, pops, adj, scenarios, p0, T)
    if B <= 1e-9:
        B = 1e-9

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print(f"F={F:.4f} B={B:.4f} doses_used={total_doses}/{budget}")
    print("Ratio: %.6f" % (sc / 1000.0))
    sys.exit(0)

if __name__ == "__main__":
    main()
