# TIER: strong
# The insight: COMMIT to the compositional physical form (Snell's ray-parameter
# invariant n(z) sin(theta(z)) = n0 sin(theta0) holding across an entire
# stratified stack) instead of collapsing the data to a single effective
# medium. Fit a genuine TWO-layer model -- one shallow "everything else"
# layer plus one deep layer that is allowed to sit BELOW n0 -- by nonlinear
# least squares against the training rows.
#
# Even though training only ever sees near-normal incidence, the cubic
# (theta^3) correction to offset/time already carries a *different* weighted
# combination of the layer indices than the linear term does. A model that
# insists on the real compositional/Snell shape (rather than one aggregate
# parameter) can exploit that extra, weak constraint to pull its deep layer's
# index down toward the true critical-layer index -- recovering an
# approximate location for the total-internal-reflection cutoff that is
# simply invisible to any model without an explicit multi-layer structure.
#
# We sweep the deep-layer index n_b on a fine deterministic grid (this is
# exactly the "index ratio the training data weakly pins down"), and for each
# candidate solve the remaining (shallow thickness, shallow index) by pattern
# search, warm-started from the previous candidate for a smooth, fast sweep.
import sys, math


def trace(theta0, n0, layers):
    s = math.sin(theta0)
    x = 0.0
    tt = 0.0
    for d, n in layers:
        sin_i = (n0 / n) * s
        if sin_i >= 1.0 - 1e-15:
            return None
        cos_i = math.sqrt(max(0.0, 1.0 - sin_i * sin_i))
        theta_i = math.asin(sin_i)
        x += d * math.tan(theta_i)
        tt += d * n / cos_i
    return x, tt


def read_rows():
    data = sys.stdin.read().split()
    n0 = float(data[0])
    D = float(data[1])
    n_train = int(data[2])
    rows = []
    idx = 4
    for _ in range(n_train):
        deg, xo, to = float(data[idx]), float(data[idx + 1]), float(data[idx + 2])
        idx += 3
        rows.append((deg, xo, to))
    return n0, D, rows


def std_or_one(vals):
    m = sum(vals) / len(vals)
    v = sum((x - m) ** 2 for x in vals) / len(vals)
    return math.sqrt(v) if v > 1e-12 else 1.0


def train_cost(n0, D, da, na, nb, rows, sx, st):
    layers = [(da, na), (D - da, nb)]
    cost = 0.0
    for deg, xo, to in rows:
        r = trace(math.radians(deg), n0, layers)
        if r is None:
            cost += 25.0
            continue
        x, tm = r
        cost += ((x - xo) / sx) ** 2 + ((tm - to) / st) ** 2
    return cost


def pattern_search(n0, D, da, na, nb, rows, sx, st, cost0):
    step_d, step_n = 1.0, 0.05
    for _round in range(10):
        improved = False
        for dd, dn in ((step_d, 0.0), (-step_d, 0.0), (0.0, step_n), (0.0, -step_n)):
            da2 = max(0.05, min(D - 0.05, da + dd))
            na2 = max(1.0, na + dn)
            c = train_cost(n0, D, da2, na2, nb, rows, sx, st)
            if c < cost0:
                cost0, da, na = c, da2, na2
                improved = True
        if not improved:
            step_d *= 0.5
            step_n *= 0.5
            if step_d < 1e-3:
                break
    return cost0, da, na


def fit_two(n0, D, rows, sx, st):
    best_overall = None
    da_warm, na_warm = D * 0.3, 1.4
    nb100 = 80
    while nb100 <= 135:
        nb = nb100 / 100.0
        candidates = [(da_warm, na_warm)]
        for da0 in (1.0, 3.0, 5.0, 7.0, 9.0):
            for na0 in (1.32, 1.40, 1.48):
                candidates.append((da0, na0))
        best_local = None
        for da0, na0 in candidates:
            da0c = max(0.05, min(D - 0.05, da0))
            c0 = train_cost(n0, D, da0c, na0, nb, rows, sx, st)
            if best_local is None or c0 < best_local[0]:
                best_local = (c0, da0c, na0)
        cost0, da, na = best_local
        cost0, da, na = pattern_search(n0, D, da, na, nb, rows, sx, st, cost0)
        da_warm, na_warm = da, na
        if best_overall is None or cost0 < best_overall[0]:
            best_overall = (cost0, da, na, nb)
        nb100 += 1
    return best_overall[1], best_overall[2], best_overall[3]


def main():
    n0, D, rows = read_rows()
    xs = [r[1] for r in rows]
    ts = [r[2] for r in rows]
    sx = std_or_one(xs)
    st = std_or_one(ts)
    da, na, nb = fit_two(n0, D, rows, sx, st)
    print(2)
    print("%.6f %.6f" % (da, na))
    print("%.6f %.6f" % (D - da, nb))


if __name__ == "__main__":
    main()
