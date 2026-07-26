# TIER: greedy
"""
The obvious, competent-looking approach: notice the training data shows a
final-color measurement that is (almost) a function of the ENDPOINT
temperature alone, with one sharp jump somewhere in the middle.  Find that
jump via a 1-D two-segment least-squares changepoint search (this easily
beats "ignore T"), fit a line on each side, and treat the changepoint as A
SINGLE threshold: below it -> cold branch value, above it -> hot branch
value.  This is a pointwise regression on the protocol's endpoint; it never
looks at the rest of the path, so it implicitly assumes T_down == T_up
(zero-width hysteresis / no branch memory).  It fits the monotone training
data essentially perfectly (path and endpoint ARE confounded there) but is
wrong every time a held-out protocol reheats into the bistable window
without crossing back over the true upper spinodal.
"""
import sys


def read_training():
    data = sys.stdin.read().split()
    N = int(data[2])
    idx = 4
    xs, ys = [], []
    for _ in range(N):
        K = int(data[idx])
        idx += 1
        proto = [float(data[idx + j]) for j in range(K)]
        idx += K
        m = float(data[idx])
        idx += 1
        xs.append(proto[-1])
        ys.append(m)
    return xs, ys


def changepoint_fit(xs, ys):
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    sx = [xs[i] for i in order]
    sy = [ys[i] for i in order]

    Px = [0.0] * (n + 1)
    Py = [0.0] * (n + 1)
    Pxx = [0.0] * (n + 1)
    Pxy = [0.0] * (n + 1)
    Pyy = [0.0] * (n + 1)
    for i in range(n):
        Px[i + 1] = Px[i] + sx[i]
        Py[i + 1] = Py[i] + sy[i]
        Pxx[i + 1] = Pxx[i] + sx[i] * sx[i]
        Pxy[i + 1] = Pxy[i] + sx[i] * sy[i]
        Pyy[i + 1] = Pyy[i] + sy[i] * sy[i]

    def fit_sse(n_pts, Sx, Sy, Sxx, Sxy, Syy):
        denom = n_pts * Sxx - Sx * Sx
        if n_pts < 2 or abs(denom) < 1e-9:
            return None
        b = (n_pts * Sxy - Sx * Sy) / denom
        a = (Sy - b * Sx) / n_pts
        sse = Syy - a * Sy - b * Sxy
        return a, b, max(0.0, sse)

    min_side = max(5, n // 20)
    best = None
    for k in range(min_side, n - min_side + 1):
        left = fit_sse(k, Px[k], Py[k], Pxx[k], Pxy[k], Pyy[k])
        right = fit_sse(n - k, Px[n] - Px[k], Py[n] - Py[k],
                         Pxx[n] - Pxx[k], Pxy[n] - Pxy[k], Pyy[n] - Pyy[k])
        if left is None or right is None:
            continue
        total = left[2] + right[2]
        if best is None or total < best[0]:
            split_T = (sx[k - 1] + sx[k]) / 2.0
            best = (total, split_T, left, right)
    if best is None:
        mean_y = sum(ys) / n
        return mean_y, 0.0, mean_y, 0.0, sum(xs) / n
    _, split_T, left, right = best
    a_cold, b_cold, _ = left
    a_hot, b_hot, _ = right
    return a_hot, b_hot, a_cold, b_cold, split_T


def main():
    xs, ys = read_training()
    a_hot, b_hot, a_cold, b_cold, T_split = changepoint_fit(xs, ys)
    T_center = 600.0
    # changepoint_fit returns raw-form (y = a + b*T); re-express in the
    # checker's A + B*(T-600) form: A = a + b*600, B = b.
    A1, B1 = a_hot + b_hot * T_center, b_hot
    A2, B2 = a_cold + b_cold * T_center, b_cold
    print("%.6f %.6f" % (A1, B1))
    print("%.6f %.6f" % (A2, B2))
    print("%.6f %.6f" % (T_split, T_split))


if __name__ == "__main__":
    main()
