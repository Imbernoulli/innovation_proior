# TIER: greedy
# The obvious recipe: the training rows only ever probe near-vertical aim, so
# fit the SIMPLEST model consistent with them -- ONE uniform slab spanning the
# whole depth D, with its refractive index n_eff chosen by least squares
# against the (offset, time) training data. This is an excellent fit on the
# near-normal training range (any smooth multi-layer stack looks like a single
# effective medium to leading order there), and it extrapolates smoothly and
# monotonically to steep angles.
#
# The trap: n_eff ends up near the *thickness-weighted harmonic mean* of the
# true layer indices, which is NEVER equal to the true deep (critical) layer's
# own index. So the single-slab model's implied critical angle (if it has one
# at all) sits far from the true cutoff -- it confidently predicts "the ray
# exits" deep into the regime where the real ray has already been totally
# internally reflected, and it never intentionally models a cutoff at all.
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
    # data[3] is the test id, skip
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


def fit_single(n0, D, rows, sx, st):
    best = None
    n = 0.60
    while n <= 3.00 + 1e-9:
        cost = 0.0
        layers = [(D, n)]
        for deg, xo, to in rows:
            r = trace(math.radians(deg), n0, layers)
            if r is None:
                cost += 25.0
            else:
                x, tm = r
                cost += ((x - xo) / sx) ** 2 + ((tm - to) / st) ** 2
        if best is None or cost < best[0]:
            best = (cost, n)
        n += 0.001
    return best[1]


def main():
    n0, D, rows = read_rows()
    xs = [r[1] for r in rows]
    ts = [r[2] for r in rows]
    sx = std_or_one(xs)
    st = std_or_one(ts)
    n_eff = fit_single(n0, D, rows, sx, st)
    print(1)
    print("%.6f %.6f" % (D, n_eff))


if __name__ == "__main__":
    main()
