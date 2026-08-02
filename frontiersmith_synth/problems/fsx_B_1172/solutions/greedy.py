# TIER: greedy
"""Textbook slope-intercept refraction interpretation. Detect the distinct
constant-slope segments of the given first-arrival curve (left to right =
shallow to deep) and read them straight off as v_1, v_2, ..., v_k in order --
the standard recipe an experienced coder reaches for first. It never checks
whether the segment count actually matches the N layers the problem asked
for: if a low-velocity zone hid one interior branch, this silently shifts
every velocity from that point on down by one index, and the last layer is
just extrapolated. It also never looks at whether the resulting thicknesses
are physically plausible."""
import sys


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
    P = len(segv)

    v = [0.0] * (N + 1)
    if P >= N:
        for i in range(1, N + 1):
            v[i] = segv[i - 1]
    else:
        for i in range(1, N):
            v[i] = segv[i - 1] if i - 1 < P else v[i - 1] * 1.15
        incs = [v[i] - v[i - 1] for i in range(2, N)]
        med_inc = sorted(incs)[len(incs) // 2] if incs else 500.0
        v[N] = v[N - 1] + med_inc

    h = [0.0] * N
    for k in range(1, N):
        h[k] = v[k] * (taus[k - 1] - (taus[k - 2] if k >= 2 else 0.0)) / 2.0

    out = []
    for k in range(1, N):
        out.append("%.6f %.6f" % (h[k], v[k]))
    out.append("%.6f" % v[N])
    print("\n".join(out))


if __name__ == "__main__":
    main()
