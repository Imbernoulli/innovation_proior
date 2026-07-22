#!/usr/bin/env python3
# verify.py <in> <out> <ans>   -- deterministic scorer for fast-pass pricing.
#
# Reads the instance, reads the participant's M prices, computes the deterministic
# self-selection equilibrium (fixed-order best-response sweeps), scores
#   score_raw = revenue + (LAM/1000) * consumer_surplus
# and normalizes against the checker's own baseline B = score_raw at the posted
# reference prices.  Prints `Ratio: <float in [0,1]>`.
#
# Equilibrium (price-taking, cardinality budget):  every visitor buys at most K_i
# passes, taking the rides in their itinerary with the largest POSITIVE net benefit
#   net_ij = v_i * Savings_j(F_j) - p_j ,   Savings_j(F) = (n_j-F)/s_reg_j - F/s_fast_j
# where F_j is the current number of buyers of ride j.  Best response is applied in a
# fixed visitor order for T sweeps (Gauss-Seidel; converges in a congestion game).
import sys, math

def read_ints(path):
    with open(path, "rb") as f:
        data = f.read().split()
    return data

def load_instance(inp):
    it = iter(inp)
    def nxt():
        return next(it)
    M = int(nxt()); N = int(nxt()); LAM = int(nxt()); T = int(nxt())
    s_reg = [0]*M; s_fast = [0]*M; ref = [0]*M
    for j in range(M):
        s_reg[j] = int(nxt()); s_fast[j] = int(nxt()); ref[j] = int(nxt())
    vval = [0]*N; kbud = [0]*N; itins = [None]*N
    for i in range(N):
        v = int(nxt()); K = int(nxt()); ks = int(nxt())
        itn = [int(nxt())-1 for _ in range(ks)]
        vval[i] = v; kbud[i] = K; itins[i] = itn
    # demand per ride
    n = [0]*M
    for i in range(N):
        for j in itins[i]:
            n[j] += 1
    return M, N, LAM, T, s_reg, s_fast, ref, vval, kbud, itins, n

def equilibrium_score(prices, M, N, LAM, T, s_reg, s_fast, vval, kbud, itins, n):
    """Return score_raw for the given price vector via fixed-order best response."""
    inv_sr = [1.0/s_reg[j] for j in range(M)]
    inv_sf = [1.0/s_fast[j] for j in range(M)]
    nf = [float(n[j]) for j in range(M)]
    F = [0]*M                     # current buyer count per ride
    bought = [[] for _ in range(N)]   # current rides each visitor buys

    def savings(j, f):
        return (nf[j]-f)*inv_sr[j] - f*inv_sf[j]

    for t in range(T):
        changed = False
        for i in range(N):
            itn = itins[i]
            if not itn:
                continue
            cur = bought[i]
            # remove this visitor's contribution
            if cur:
                for j in cur:
                    F[j] -= 1
            v = vval[i]
            # candidate net benefit for each ride if THIS visitor joins it
            cands = []
            for j in itn:
                p = prices[j]
                sv = savings(j, F[j] + 1)      # savings after i joins
                net = v * sv - p
                if net > 0.0:
                    cands.append((net, j))
            if cands:
                cands.sort(reverse=True)
                take = cands[:kbud[i]]
                newset = [j for (_, j) in take]
            else:
                newset = []
            for j in newset:
                F[j] += 1
            if newset != cur:
                changed = True
            bought[i] = newset
        if not changed:
            break

    # ---- realized score ----
    # sum of buyer time-values per ride (for surplus)
    sumv = [0.0]*M
    for i in range(N):
        v = vval[i]
        for j in bought[i]:
            sumv[j] += v
    lam = LAM / 1000.0
    revenue = 0.0
    surplus = 0.0
    for j in range(M):
        f = F[j]
        if f <= 0:
            continue
        sv = savings(j, f)                    # realized savings at equilibrium
        revenue += prices[j] * f
        surplus += sumv[j]*sv - prices[j]*f   # realized net surplus of buyers
    return revenue + lam * surplus

def fail(msg):
    print("Ratio: 0.0  (%s)" % msg)
    sys.exit(0)

def main():
    inpath, outpath = sys.argv[1], sys.argv[2]
    inst = load_instance(read_ints(inpath))
    (M, N, LAM, T, s_reg, s_fast, ref, vval, kbud, itins, n) = inst

    # ---- parse participant prices ----
    try:
        with open(outpath, "r") as f:
            toks = f.read().split()
    except Exception:
        fail("no output")
    if len(toks) < M:
        fail("expected %d prices, got %d" % (M, len(toks)))
    prices = []
    for t in toks[:M]:
        try:
            x = float(t)
        except ValueError:
            fail("non-numeric price")
        if not math.isfinite(x):
            fail("non-finite price")
        if x < 0.0:
            fail("negative price")
        if x > 1e12:
            fail("price out of range")
        prices.append(x)
    if len(toks) > M + 2:
        fail("too many tokens")

    sub = equilibrium_score(prices, M, N, LAM, T, s_reg, s_fast, vval, kbud, itins, n)
    base = equilibrium_score([float(r) for r in ref], M, N, LAM, T, s_reg, s_fast, vval, kbud, itins, n)
    B = max(1e-9, base)
    sc = min(1000.0, 100.0 * sub / B)
    if sc < 0.0:
        sc = 0.0
    ratio = sc / 1000.0
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0:
        ratio = 1.0
    print("revenue-surplus score=%.4f baseline=%.4f Ratio: %.6f" % (sub, base, ratio))

if __name__ == "__main__":
    main()
