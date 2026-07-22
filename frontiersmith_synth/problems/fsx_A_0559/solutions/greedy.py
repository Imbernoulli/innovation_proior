# TIER: greedy
# The obvious textbook approach: a monotone PROGRESSIVE schedule.
# Thresholds at income percentiles, marginal rates rising with income.
# It ignores elasticity entirely, so it over-taxes the elastic middle band.
# If the welfare floor is violated, it repairs by shaving the top rate down.
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


def evaluate(m, e, f, thresholds, rates):
    K = len(rates)
    thr = np.asarray(thresholds, float); rt = np.asarray(rates, float)
    inv = 1.0 + 1.0 / e
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
    return float(np.sum(np.where(works, tax_of(best_z), 0.0))), float(np.sum(np.where(works, up, 0.0)))


def main():
    m, e, f, floor = read_pop()
    z0 = m  # income at zero rate as a proxy for position
    qs = [20, 40, 60, 80]
    thresholds = [0.0] + [float(np.percentile(z0, q)) for q in qs]
    # dedupe/strictly increase
    th = [thresholds[0]]
    for t in thresholds[1:]:
        if t > th[-1] + 1e-6:
            th.append(t)
    thresholds = th
    K = len(thresholds)
    rates = [min(RATE_MAX, 0.15 + 0.16 * k) for k in range(K)]  # rising 0.15,0.31,...

    R, W = evaluate(m, e, f, thresholds, rates)
    # repair: shave the highest rates until the floor is met
    guard = 0
    while W < floor and guard < 200:
        j = int(np.argmax(rates))
        rates[j] = max(0.0, rates[j] - 0.02)
        R, W = evaluate(m, e, f, thresholds, rates)
        guard += 1

    print(K)
    for b, r in zip(thresholds, rates):
        print("%.6f %.6f" % (b, r))


main()
