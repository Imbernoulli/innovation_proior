# TIER: strong
"""Reflection/refraction cross-check that exploits the velocity-depth trade-off
instead of falling for it.

Insight: the reflection two-way times tau_k exist for EVERY interface, hidden
or not, and pin each layer's h_j/v_j ratio exactly. The refraction curve's
segments give every velocity EXCEPT one if a low-velocity zone hid a branch
(reflection sees it, refraction doesn't -- that asymmetry is the tell). When
the segment count is one short of N, this solution does not just shift
indices and hope: it enumerates every possible position for the missing
layer, and for each candidate checks whether the OTHER N-2 thicknesses (from
tau ratios with the hypothesized velocity labelling) are physically sane.
Among candidates that pass, it 1-D-searches the single unknown velocity to
minimize the residual between the resulting model's predicted refraction
curve and the refraction data actually given -- exploiting the fact that the
hidden layer still delays every deeper head wave even though it never
produces one of its own. The candidate/velocity pair with the smallest
residual is the reconstruction."""
import sys

H_MIN, H_MAX = 20.0, 400.0


def first_arrival(x, h, v, N):
    best = x / v[1]
    running_max = v[1]
    for i in range(1, N):
        vi1 = v[i + 1]
        if vi1 > running_max:
            c = 0.0
            for j in range(1, i + 1):
                vj = v[j]
                diff = vi1 * vi1 - vj * vj
                if diff > 0:
                    c += h[j] * (diff ** 0.5) / (vj * vi1)
            t = x / vi1 + 2.0 * c
            if t < best:
                best = t
        running_max = max(running_max, vi1)
    return best


def detect_segments(xs, ts):
    slopes = [(ts[i + 1] - ts[i]) / (xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
    clusters = []
    cur = [slopes[0]]
    for i in range(1, len(slopes)):
        ref = sorted(cur)[len(cur) // 2]
        if abs(slopes[i] - ref) / ref > 0.015:
            clusters.append(cur)
            cur = [slopes[i]]
        else:
            cur.append(slopes[i])
    clusters.append(cur)
    changed = True
    while changed and len(clusters) > 1:
        changed = False
        for idx, c in enumerate(clusters):
            if len(c) < 2:
                cands = []
                if idx > 0:
                    cands.append(idx - 1)
                if idx < len(clusters) - 1:
                    cands.append(idx + 1)
                medc = c[0]
                best = min(cands, key=lambda j: abs(sorted(clusters[j])[len(clusters[j]) // 2] - medc))
                clusters[best] = clusters[best] + c
                del clusters[idx]
                changed = True
                break
    return [1.0 / (sum(c) / len(c)) for c in clusters]


def greedy_fallback(N, segv, taus):
    P = len(segv)
    v = [0.0] * (N + 1)
    for i in range(1, N):
        v[i] = segv[i - 1] if i - 1 < P else v[i - 1] * 1.15
    v[N] = v[N - 1] * 1.15
    h = [0.0] * N
    for k in range(1, N):
        h[k] = v[k] * (taus[k - 1] - (taus[k - 2] if k >= 2 else 0.0)) / 2.0
    return h, v


def solve(N, segv, taus, xs, ts):
    P = len(segv)
    if P >= N:
        v = [0.0] * (N + 1)
        for i in range(1, N + 1):
            v[i] = segv[i - 1]
        h = [0.0] * N
        for k in range(1, N):
            h[k] = v[k] * (taus[k - 1] - (taus[k - 2] if k >= 2 else 0.0)) / 2.0
        return h, v

    best_m, best_resid, best_vm = None, None, None
    for m in range(2, N):
        v = [0.0] * (N + 1)
        for i in range(1, m):
            v[i] = segv[i - 1]
        for i in range(m + 1, N + 1):
            v[i] = segv[i - 2]
        ok = True
        h = [0.0] * N
        for k in range(1, N):
            if k == m:
                continue
            h[k] = v[k] * (taus[k - 1] - (taus[k - 2] if k >= 2 else 0.0)) / 2.0
            if not (H_MIN - 1e-3 <= h[k] <= H_MAX + 1e-3):
                ok = False
        if not ok:
            continue
        r_m = (taus[m - 1] - (taus[m - 2] if m >= 2 else 0.0)) / 2.0  # h_m / v_m, exact from reflection data
        vlo, vhi = 1.0, v[m - 1] * 0.999
        if vhi <= vlo:
            continue

        def resid(vm):
            h[m] = r_m * vm
            v[m] = vm
            s = 0.0
            for x, t in zip(xs, ts):
                tp = first_arrival(x, h, v, N)
                s += (tp - t) ** 2
            return s

        NG = 80
        cvals = [vlo + (vhi - vlo) * i / (NG - 1) for i in range(NG)]
        best_local = min(cvals, key=resid)
        lo2 = max(vlo, best_local - (vhi - vlo) / NG * 2)
        hi2 = min(vhi, best_local + (vhi - vlo) / NG * 2)
        for _ in range(25):
            a = lo2 + (hi2 - lo2) / 3
            b = hi2 - (hi2 - lo2) / 3
            if resid(a) < resid(b):
                hi2 = b
            else:
                lo2 = a
        vm_star = (lo2 + hi2) / 2
        r = resid(vm_star)
        if best_resid is None or r < best_resid:
            best_resid, best_m, best_vm = r, m, vm_star

    if best_m is None:
        return greedy_fallback(N, segv, taus)

    m = best_m
    v = [0.0] * (N + 1)
    for i in range(1, m):
        v[i] = segv[i - 1]
    for i in range(m + 1, N + 1):
        v[i] = segv[i - 2]
    v[m] = best_vm
    r_m = (taus[m - 1] - (taus[m - 2] if m >= 2 else 0.0)) / 2.0
    h = [0.0] * N
    for k in range(1, N):
        h[k] = r_m * v[m] if k == m else v[k] * (taus[k - 1] - (taus[k - 2] if k >= 2 else 0.0)) / 2.0
    return h, v


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    test_id = int(next(it))
    N = int(next(it))
    M = int(next(it))
    xs, ts = [], []
    for _ in range(M):
        xs.append(float(next(it)))
        ts.append(float(next(it)))
    taus = [float(next(it)) for _ in range(N - 1)]

    segv = detect_segments(xs, ts)
    h, v = solve(N, segv, taus, xs, ts)

    out = []
    for k in range(1, N):
        out.append("%.6f %.6f" % (h[k], v[k]))
    out.append("%.6f" % v[N])
    print("\n".join(out))


if __name__ == "__main__":
    main()
