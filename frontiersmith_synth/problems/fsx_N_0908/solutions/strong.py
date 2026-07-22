# TIER: strong
"""The insight: because the exact realization of tactic shifts each round is unpredictable
(only the migration GRAPH is public, not the timing/target of any given hop), a plan that
commits its ENTIRE budget to precisely matching the noise-free forecast is exposed every
round reality deviates from that forecast. Reserve a substantial, STEADY floor of the budget
-- weighted by where the migration graph structurally routes pressure (power-iterate M,
weighted by harm value), not by this round's forecast volume -- so coverage is already
sitting on the categories the graph says are generically attractive, regardless of which
one happens to spike this round. Spend the remainder efficiently on the forecast, and blend
round to round so the plan does not lurch to chase every forecast wiggle.

This is not "greedy plus a tuned exponent": the floor is a FIXED profile derived purely from
the public migration graph and per-category value -- it never looks at the round's forecast
volume at all -- and it is what buys robustness to the unpredictable hop the forecast-only
plan (greedy tier) cannot hedge against."""
import sys, json


def structural_weight(K, M, value, iters=25):
    w = [1.0 / K] * K
    for _ in range(iters):
        nw = [0.0] * K
        for i in range(K):
            Mi = M[i]
            for j in range(K):
                if Mi[j] > 0:
                    nw[j] += w[i] * Mi[j]
        s = sum(nw)
        nw = [x / s for x in nw] if s > 1e-12 else [1.0 / K] * K
        w = [0.85 * a + 0.15 * (1.0 / K) for a in nw]
        s = sum(w)
        w = [x / s for x in w]
    ww = [w[j] * value[j] for j in range(K)]
    s = sum(ww)
    return [x / s for x in ww]


def main():
    inst = json.load(sys.stdin)
    K, T = inst["K"], inst["T"]
    V, R, value, M, beta = inst["V"], inst["R"], inst["value"], inst["M"], inst["beta"]
    floor_frac, smooth = 0.8, 0.6

    w = structural_weight(K, M, value)
    p = list(inst["p0"])
    prev = [0.0] * K
    alloc = []
    for t in range(T):
        Vt, Rt = V[t], R[t]
        vol = [Vt * p[j] for j in range(K)]

        raw_w = [max(vol[j], 1e-9) * value[j] for j in range(K)]     # reactive forecast core
        s = sum(raw_w)
        raw = [Rt * x / s for x in raw_w]

        sf = [Rt * w[j] for j in range(K)]                           # structural floor
        raw = [(1 - floor_frac) * raw[j] + floor_frac * sf[j] for j in range(K)]

        if t and smooth > 0:                                          # avoid lurching
            row = [smooth * prev[j] * Rt / R[t - 1] + (1 - smooth) * raw[j] for j in range(K)]
        else:
            row = raw
        s = sum(row)
        row = [x * Rt / s for x in row]
        alloc.append(row)

        cov = [0.0 if vol[j] <= 1e-12 else min(1.0, row[j] / vol[j]) for j in range(K)]
        newp = [0.0] * K
        for i in range(K):
            stay = p[i] * (1 - beta * cov[i])
            newp[i] += stay
            leave = p[i] * beta * cov[i]
            if leave > 0:
                Mi = M[i]
                for j in range(K):
                    if j != i and Mi[j] > 0:
                        newp[j] += leave * Mi[j]
        p = newp
        prev = list(row)

    print(json.dumps({"alloc": alloc}))


main()
