# TIER: strong
# Insight (two parts, both about reading the CONSISTENCY GRAPH rather than any single trace):
#
# 1. Predict a sensor from the MEDIAN (not mean) of its direct neighbours. A lying neighbour
#    contaminates a mean-based prediction for everyone around it (one bad reading drags every
#    neighbour's residual toward a spurious "persistent bias" and manufactures false positives
#    in a ring around the real fault) -- the median shrugs off a single bad neighbour instead.
#
# 2. Peel faults off ONE AT A TIME: find the sensor whose neighbour-residual is best explained
#    by a persistent straight line (offset+drift) versus a real local event, which stays a
#    short bump a line fits poorly even after light trimming; declare only the single most
#    confident case, SUBTRACT its estimated correction from the working copy, and re-derive
#    every neighbour prediction from the now-decontaminated readings before looking for the
#    next fault. This stops one true fault from posing as false evidence against its neighbours,
#    and lets the algorithm still recover a sensor that carries BOTH a real fault and a
#    genuine co-located event.
import sys


def ols_fit(ts, ys):
    m = len(ts)
    if m == 0:
        return 0.0, 0.0
    st = sum(ts)
    sy = sum(ys)
    stt = sum(t * t for t in ts)
    sty = sum(t * y for t, y in zip(ts, ys))
    denom = m * stt - st * st
    if abs(denom) < 1e-9:
        return sy / m, 0.0
    b = (m * sty - st * sy) / denom
    a = (sy - b * st) / m
    return a, b


def robust_fit(ts, ys, trim_frac=0.15):
    a0, b0 = ols_fit(ts, ys)
    resid = [abs(y - (a0 + b0 * t)) for t, y in zip(ts, ys)]
    order = sorted(range(len(ts)), key=lambda k: resid[k])
    keep_n = max(2, int(round(len(ts) * (1.0 - trim_frac))))
    keep = sorted(order[:keep_n])
    kt = [ts[k] for k in keep]
    ky = [ys[k] for k in keep]
    a1, b1 = ols_fit(kt, ky)
    sse_lin = sum((y - (a1 + b1 * t)) ** 2 for t, y in zip(kt, ky))
    sse_null = sum(y * y for y in ky)
    r2 = 1.0 - sse_lin / sse_null if sse_null > 1e-9 else 0.0
    return a1, b1, r2


def median(xs):
    s = sorted(xs)
    m = len(s)
    if m % 2 == 1:
        return s[m // 2]
    return 0.5 * (s[m // 2 - 1] + s[m // 2])


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    test_id = int(next(it))
    n = int(next(it)); T = int(next(it)); F_max = int(next(it))
    m_edges = int(next(it))
    adj = [[] for _ in range(n)]
    for _ in range(m_edges):
        u = int(next(it)); v = int(next(it))
        adj[u].append(v)
        adj[v].append(u)
    R = [[0.0] * T for _ in range(n)]
    for i in range(n):
        for t in range(T):
            R[i][t] = float(next(it))

    ts = list(range(T))
    work = [row[:] for row in R]                # decontaminated working copy
    declared_order = []
    declared_fit = {}
    thresh = 0.55

    for _ in range(F_max):
        best_i, best_r2, best_a, best_b = -1, thresh, 0.0, 0.0
        for i in range(n):
            if i in declared_fit:
                continue
            nb = adj[i]
            if nb:
                m_t = [median([work[j][t] for j in nb]) for t in range(T)]
            else:
                m_t = [0.0] * T
            e = [work[i][t] - m_t[t] for t in range(T)]
            a, b, r2 = robust_fit(ts, e)
            if r2 > best_r2:
                best_i, best_r2, best_a, best_b = i, r2, a, b
        if best_i < 0:
            break
        declared_fit[best_i] = (best_a, best_b)
        declared_order.append(best_i)
        for t in range(T):
            work[best_i][t] = R[best_i][t] - best_a - best_b * t

    out = [str(len(declared_order))]
    for i in declared_order:
        a, b = declared_fit[i]
        out.append(f"{i} {a:.6f} {b:.6f}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
