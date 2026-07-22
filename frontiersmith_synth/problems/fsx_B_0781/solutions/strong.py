# TIER: strong
"""The insight: split the training sweep into its monotone RUNS (direction
segments). Within the back half of each run the backlash has already locked
onto its edge, so a per-run affine fit y = a_run*x + b_run gives an UNBIASED
estimate of the ratio a_run =~ r (the constant lag is absorbed cleanly into
b_run, per-run, instead of being smeared across a single global intercept).
Average those per-run slopes, then -- the key identification step -- SNAP
the averaged slope to the nearest small rational p/q (small gear tooth
counts): this separates the discrete ratio from the continuous backlash bias
that would otherwise stay confounded in a plain real-valued slope estimate.
Once r is pinned down, each run's intercept directly reveals the backlash
half-width D = |b_run| / r. Emit the exact backlash PLAY OPERATOR (clip the
previous contact into the current [x-D, x+D] band) times the recovered
rational ratio -- a stateful program that, unlike any memoryless curve,
tracks which edge is currently engaged and so generalises to a fast,
many-reversal drive."""
import sys


def _sign(v):
    if v > 1e-9:
        return 1
    if v < -1e-9:
        return -1
    return 0


def _ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx if sxx > 1e-9 else 0.0
    b = my - a * mx
    return a, b


def _find_runs(xs):
    n = len(xs)
    dirs = [_sign(xs[i] - xs[i - 1]) for i in range(1, n)]
    runs = []
    start = 0
    cur = None
    for i, d in enumerate(dirs):
        if d == 0:
            continue
        if cur is None:
            cur, start = d, i
        elif d != cur:
            runs.append((start, i + 1))
            cur, start = d, i
    runs.append((start, n))
    return runs


def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    vals = list(map(float, data[2:2 + 2 * n]))
    xs = vals[0::2]
    ys = vals[1::2]

    runs = _find_runs(xs)
    slopes, intercepts, weights = [], [], []
    for (s, e) in runs:
        L = e - s
        if L < 8:
            continue
        s2 = s + int(L * 0.4)  # drop the first 40% -- possible backlash-transit transient
        sub_x, sub_y = xs[s2:e], ys[s2:e]
        if len(sub_x) < 4:
            continue
        a, b = _ols(sub_x, sub_y)
        slopes.append(a)
        intercepts.append(b)
        weights.append(len(sub_x))

    if slopes:
        tot = sum(weights)
        r_est = sum(a * w for a, w in zip(slopes, weights)) / tot
    else:
        r_est = 1.0

    best = None
    for q in range(1, 10):
        for p in range(1, 10):
            v = p / q
            dd = abs(v - r_est)
            if best is None or dd < best[0]:
                best = (dd, p, q)
    r_hat = best[1] / best[2]

    if intercepts:
        d_vals = [abs(b) / r_hat if r_hat > 1e-6 else 0.0 for b in intercepts]
        D_hat = sum(dv * w for dv, w in zip(d_vals, weights)) / sum(weights)
    else:
        D_hat = 0.2
    D_hat = max(0.01, D_hat)

    print("STATE max2( min2( Sk1, x + %.6f ), x - %.6f )" % (D_hat, D_hat))
    print("OUT   %.6f * S" % r_hat)


if __name__ == "__main__":
    main()
