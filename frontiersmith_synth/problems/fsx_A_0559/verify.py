import sys
import numpy as np

# Deterministic scorer for the laffer-ledge-tax family.
#   objective: MAXIMIZE tax revenue subject to an aggregate-welfare floor.
# Participant output = a piecewise-linear marginal-rate schedule (<=8 brackets).

RATE_MAX = 0.95
MAX_BRACKETS = 8
THRESH_MAX = 1.0e7
BASE_RATE = 0.06   # checker's internal baseline (a mild flat tax)


def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)


def evaluate(m, e, f, thresholds, rates):
    K = len(rates)
    thr = np.asarray(thresholds, dtype=float)
    rt = np.asarray(rates, dtype=float)
    inv = 1.0 + 1.0 / e
    Tcum = np.zeros(K)
    for k in range(1, K):
        Tcum[k] = Tcum[k - 1] + rt[k - 1] * (thr[k] - thr[k - 1])
    edges = np.concatenate([thr, [np.inf]])

    def tax_of(z):
        idx = np.searchsorted(edges, z, side='right') - 1
        idx = np.clip(idx, 0, K - 1)
        return Tcum[idx] + rt[idx] * (z - thr[idx])

    def util(z):
        v = (m / inv) * np.power(z / m, inv)
        return (z - tax_of(z)) - v

    best_u = np.zeros_like(m)
    best_z = np.zeros_like(m)
    have = np.zeros(len(m), dtype=bool)

    def consider(z, valid):
        nonlocal best_u, best_z, have
        u = util(z)
        take = valid & (~have | (u > best_u + 1e-12))
        best_u = np.where(take, u, best_u)
        best_z = np.where(take, z, best_z)
        have = have | valid

    for k in range(K):
        zc = m * np.power(1.0 - rt[k], e)
        hi = edges[k + 1]
        valid = (zc >= thr[k]) & (zc <= hi) & (zc > 0)
        consider(zc, valid)
    for k in range(1, K):
        zc = np.full_like(m, thr[k])
        consider(zc, zc > 0)

    up = best_u - f
    works = have & (up > 1e-12)
    R = float(np.sum(np.where(works, tax_of(best_z), 0.0)))
    W = float(np.sum(np.where(works, up, 0.0)))
    return R, W


def main():
    inp = open(sys.argv[1]).read().split()
    out_tokens = open(sys.argv[2]).read().split()

    try:
        it = iter(inp)
        N = int(next(it))
        floor = float(next(it))
        m = np.empty(N); e = np.empty(N); f = np.empty(N)
        for i in range(N):
            m[i] = float(next(it)); e[i] = float(next(it)); f[i] = float(next(it))
    except Exception:
        fail("bad input")

    # ---- parse participant schedule ----
    try:
        oit = iter(out_tokens)
        K = int(next(oit))
    except Exception:
        fail("parse K")
    if not (1 <= K <= MAX_BRACKETS):
        fail("K out of range")

    thresholds = []
    rates = []
    try:
        for _ in range(K):
            b = float(next(oit))
            r = float(next(oit))
            thresholds.append(b)
            rates.append(r)
    except Exception:
        fail("parse brackets")

    # reject any trailing garbage (strict token count)
    extra = list(oit)
    if extra:
        fail("trailing tokens")

    for x in thresholds + rates:
        if not np.isfinite(x):
            fail("non-finite")
    if abs(thresholds[0]) > 1e-9:
        fail("first threshold must be 0")
    for k in range(1, K):
        if not (thresholds[k] > thresholds[k - 1]):
            fail("thresholds not strictly increasing")
        if thresholds[k] > THRESH_MAX:
            fail("threshold too large")
    for r in rates:
        if not (0.0 <= r <= RATE_MAX):
            fail("rate out of range")

    R, W = evaluate(m, e, f, thresholds, rates)
    if not np.isfinite(R) or not np.isfinite(W):
        fail("non-finite objective")
    if W < floor - 1e-6:
        fail("welfare floor violated")

    # internal baseline B = revenue of a mild flat tax
    B, _ = evaluate(m, e, f, [0.0], [BASE_RATE])
    B = max(1e-9, B)

    sc = min(1000.0, 100.0 * R / B)
    print("R=%.3f B=%.3f W=%.3f floor=%.3f Ratio: %.6f" % (R, B, W, floor, sc / 1000.0))


if __name__ == "__main__":
    main()
