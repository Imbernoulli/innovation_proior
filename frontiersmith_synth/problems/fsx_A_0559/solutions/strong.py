# TIER: strong
# Insight: optimize the marginal rates against the ACTUAL population best-response,
# not a template.  Coordinate-ascent on revenue subject to the welfare floor lets
# each bracket's rate settle near its own local Laffer point -- so the bracket that
# holds the elastic mass gets a LOW rate while inelastic brackets are taxed hard.
# The resulting schedule is NON-MONOTONE (a dip where the elastic middle sits),
# a shape no progressive/flat/linear template produces.  Local search only -> the
# score ceiling stays open above this reference.
import sys
import numpy as np

RATE_MAX = 0.95


def read_pop():
    d = sys.stdin.buffer.read().split()
    it = iter(d)
    N = int(next(it)); floor = float(next(it))
    m = np.empty(N); e = np.empty(N); f = np.empty(N)
    for i in range(N):
        m[i] = float(next(it)); e[i] = float(next(it)); f[i] = float(next(it))
    return m, e, f, floor


def make_eval(m, e, f):
    inv = 1.0 + 1.0 / e

    def evaluate(thresholds, rates):
        K = len(rates)
        thr = np.asarray(thresholds, float); rt = np.asarray(rates, float)
        Tcum = np.zeros(K)
        for k in range(1, K):
            Tcum[k] = Tcum[k - 1] + rt[k - 1] * (thr[k] - thr[k - 1])
        edges = np.concatenate([thr, [np.inf]])

        def tax_of(z):
            idx = np.clip(np.searchsorted(edges, z, side='right') - 1, 0, K - 1)
            return Tcum[idx] + rt[idx] * (z - thr[idx])

        def util(z):
            return (z - tax_of(z)) - (m / inv) * np.power(z / m, inv)

        best_u = np.zeros_like(m); best_z = np.zeros_like(m)
        have = np.zeros(len(m), bool)

        def consider(z, valid):
            nonlocal best_u, best_z, have
            u = util(z)
            take = valid & (~have | (u > best_u + 1e-12))
            best_u = np.where(take, u, best_u); best_z = np.where(take, z, best_z)
            have = have | valid

        for k in range(K):
            zc = m * np.power(1.0 - rt[k], e)
            valid = (zc >= thr[k]) & (zc <= edges[k + 1]) & (zc > 0)
            consider(zc, valid)
        for k in range(1, K):
            consider(np.full_like(m, thr[k]), np.full(len(m), thr[k] > 0))
        up = best_u - f
        works = have & (up > 1e-12)
        R = float(np.sum(np.where(works, tax_of(best_z), 0.0)))
        W = float(np.sum(np.where(works, up, 0.0)))
        return R, W

    return evaluate


def main():
    m, e, f, floor = read_pop()

    # subsample for fast optimization; final schedule is scored on the full set.
    N = len(m)
    cap = 4000
    if N > cap:
        idx = np.linspace(0, N - 1, cap).astype(int)
        ms, es, fs = m[idx], e[idx], f[idx]
        scale = N / float(cap)
    else:
        ms, es, fs = m, e, f
        scale = 1.0
    floor_s = floor / scale
    ev = make_eval(ms, es, fs)

    z0 = ms
    qs = [12, 24, 36, 48, 60, 74, 87]
    thresholds = [0.0]
    for q in qs:
        t = float(np.percentile(z0, q))
        if t > thresholds[-1] + 1e-6:
            thresholds.append(t)
    thresholds = thresholds[:8]
    K = len(thresholds)

    rates = [0.10] * K  # start feasible (mild, high welfare)
    grid = [i / 100.0 for i in range(0, 96, 3)]  # 0.00 .. 0.93

    def feasible_R(rts):
        R, W = ev(thresholds, rts)
        return (R, W) if W >= floor_s else (None, W)

    for _ in range(4):
        improved = False
        for k in range(K):
            bestr = rates[k]
            bestR, _ = ev(thresholds, rates)
            for g in grid:
                trial = list(rates); trial[k] = g
                R, W = ev(thresholds, trial)
                if W >= floor_s and R > bestR + 1e-9:
                    bestR = R; bestr = g
            if abs(bestr - rates[k]) > 1e-12:
                rates[k] = bestr; improved = True
        if not improved:
            break

    # final feasibility guaranteed on the FULL population (subsample estimate of
    # the floor can be optimistic by ~1%, so repair against the exact floor here)
    ev_full = make_eval(m, e, f)
    R, W = ev_full(thresholds, rates)
    guard = 0
    while W < floor + 1e-6 and guard < 400:
        j = int(np.argmax(rates))
        rates[j] = max(0.0, rates[j] - 0.01)
        R, W = ev_full(thresholds, rates)
        guard += 1

    print(K)
    for b, r in zip(thresholds, rates):
        print("%.6f %.6f" % (b, min(RATE_MAX, r)))


main()
