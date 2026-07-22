# TIER: strong
# INSIGHT: the intensity chosen at a lap only pays its wear penalty on the
# REMAINING laps of the current stint (a pit resets wear), so effort should NOT
# be uniform -- it should ramp WITHIN a stint (coast early, push late) and the
# pit boundaries should be chosen jointly.  With resets the effective wear cost
# is the lower convex envelope of the wear curve, so the optimal pace is
# bang-bang / sawtooth rather than a smooth constant.
#
# Implementation: for each candidate number of stints, split the race into
# near-equal stints; solve each stint EXACTLY by dynamic programming over
# quantized cumulative wear (choosing the per-lap intensity ramp); keep the best
# stint count.  This strictly dominates any single constant pace.
import sys

QUANT = 0.5

def solve_stint(m, base, a, p, b, q, P, grid, wpow):
    # laps 0..m-1; state = quantized cumulative wear at start of lap.
    k = len(grid)
    maxw = m * max(wpow) + 1.0
    NB = int(maxw / QUANT) + 2
    INF = float("inf")
    dp = [0.0] * NB          # dp[wq] for the "past the end" layer
    nxt = [0.0] * NB
    choice = [[0] * NB for _ in range(m)]
    for j in range(m - 1, -1, -1):
        cj = choice[j]
        for wq in range(NB):
            w = wq * QUANT
            wpen = base + a * (w ** p)
            bestc = INF; bestx = 0
            for xi in range(k):
                nwq = int(round((w + wpow[xi]) / QUANT))
                if nwq >= NB:
                    nwq = NB - 1
                c = wpen + b / grid[xi] + dp[nwq]
                if c < bestc:
                    bestc = c; bestx = xi
            nxt[wq] = bestc; cj[wq] = bestx
        dp, nxt = nxt, dp
    # reconstruct from wear 0
    xs = []
    wq = 0
    cost = dp[0]
    for j in range(m):
        xi = choice[j][wq]
        xs.append(xi)
        wq = int(round((wq * QUANT + wpow[xi]) / QUANT))
        if wq >= NB:
            wq = NB - 1
    return cost, xs

def main():
    d = sys.stdin.read().split()
    it = iter(d)
    L = int(next(it)); k = int(next(it))
    base = float(next(it)); a = float(next(it)); p = float(next(it))
    b = float(next(it)); q = float(next(it)); P = float(next(it))
    grid = [float(next(it)) for _ in range(k)]
    wpow = [g ** q for g in grid]

    best = None
    cache = {}
    for nst in range(1, L + 1):
        base_len = L // nst
        if base_len == 0:
            break
        rem = L % nst
        lens = [base_len + (1 if s < rem else 0) for s in range(nst)]
        xs = []; pits = []; tot = (nst - 1) * P
        ok = True
        for si, m in enumerate(lens):
            if m in cache:
                sc, sxs = cache[m]
            else:
                sc, sxs = solve_stint(m, base, a, p, b, q, P, grid, wpow)
                cache[m] = (sc, sxs)
            tot += sc
            pits.append(1 if si > 0 else 0)
            pits.extend([0] * (m - 1))
            xs.extend(sxs)
        if best is None or tot < best[0]:
            best = (tot, xs, pits)

    _, xs, pits = best
    out = ["%d %d" % (xs[i], pits[i]) for i in range(L)]
    sys.stdout.write("\n".join(out) + "\n")

main()
