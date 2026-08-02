# TIER: strong
# The insight: a floor makes the log-log curve BEND UPWARD (curvature) as n
# grows -- a pure power law would stay a straight line. So (a) posit a
# floor+power-law model err = F + A*n^(-alpha), and (b) recognise that the
# visible range may mix an early steeper regime with the true late one, so
# the model must be fit on a SELF-CONSISTENT subset of the largest-n points
# rather than blindly on everything.
#
# For a candidate floor-free exponent alpha, y = F + A*n^(-alpha) is LINEAR
# in x = n^(-alpha): profile alpha out by grid search, and at each alpha do
# an ordinary linear regression of y on x for the best F, A (closed form,
# no external libraries). Try this on every contiguous "largest-n" suffix
# of the training points (at least a bit more than half of them, so a small
# early-regime contamination cannot masquerade as a perfect few-point fit),
# and keep the suffix whose (F, A, alpha) fit has the smallest per-point
# residual -- i.e. the curvature signature that is CLEANEST once the early
# burst points are dropped. That fit extrapolates because a floor+power-law
# recovered from genuinely-asymptotic points stays valid arbitrarily far out.
import sys


def fit_floor_power(ns, ys):
    best = None
    ai = 5
    while ai <= 180:
        alpha = ai / 100.0
        xs = [n ** (-alpha) for n in ns]
        nA = len(xs)
        mx = sum(xs) / nA
        my = sum(ys) / nA
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        ai += 1
        if sxx < 1e-15:
            continue
        A = sxy / sxx
        F = my - A * mx
        if F < 0 or A <= 0:
            continue
        sse = sum((F + A * x - y) ** 2 for x, y in zip(xs, ys))
        if best is None or sse < best[0]:
            best = (sse, F, A, alpha)
    if best is None:
        return 0.0, 1.0, 0.4
    return best[1], best[2], best[3]


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("0.5")
        return
    m = int(data[0])
    vals = data[2:]
    ns = [float(vals[2 * i]) for i in range(m)]
    ys = [float(vals[2 * i + 1]) for i in range(m)]
    order = sorted(range(m), key=lambda i: ns[i])
    ns = [ns[i] for i in order]
    ys = [ys[i] for i in order]

    min_sz = max(4, -(-int(0.55 * m) // 1))  # ceil(0.55*m), never below 4
    best = None
    for s in range(0, m - min_sz + 1):
        sub_n = ns[s:]
        sub_y = ys[s:]
        F, A, alpha = fit_floor_power(sub_n, sub_y)
        sse = sum((F + A * (n ** (-alpha)) - y) ** 2 for n, y in zip(sub_n, sub_y))
        per = sse / len(sub_n)
        if best is None or per < best[0]:
            best = (per, F, A, alpha)

    _, F, A, alpha = best
    print("%.8f + %.8f * n ** (%.8f)" % (F, A, -alpha))


if __name__ == "__main__":
    main()
