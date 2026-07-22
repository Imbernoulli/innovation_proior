# TIER: strong
# Structural recovery via a REFORMULATION, not "more grid points": the
# hidden law is a pure power of an EFFECTIVE COMBINED LOAD
# u = x0 + w1*x1 + w2*x2 with a SHARED exponent k. Fitting (b, k) directly
# by nonlinear least squares on y is ill-conditioned (many (b,k) combos
# trade off almost losslessly over a bounded light-traffic range). The
# insight: take LOGS. For any fixed (w1, w2), log(y) = log(b) + k*log(u) is
# an ORDINARY LINEAR regression in log-space -- well conditioned, gives k
# directly as the slope. So only the two coupling weights (w1, w2) need a
# search; for each candidate the exponent and scale drop out in closed
# form. This exploits the multi-link coupling structure (the search only
# has to find w1, w2 -- the shape of the law is fixed by the log-log
# reformulation) instead of blindly curve-fitting the visible light-traffic
# regime the way the linear recipe (greedy) does.
import sys
import math


def loglog_fit(rows, w1, w2):
    xs, ys = [], []
    for x0, x1, x2, y in rows:
        u = x0 + w1 * x1 + w2 * x2
        if y <= 1e-12 or u <= 1e-12:
            continue
        xs.append(math.log(u))
        ys.append(math.log(y))
    n = len(xs)
    if n < 5:
        return None
    sx = sum(xs); sy = sum(ys)
    sxx = sum(v * v for v in xs)
    sxy = sum(xs[i] * ys[i] for i in range(n))
    det = n * sxx - sx * sx
    if abs(det) < 1e-12:
        return None
    k = (n * sxy - sx * sy) / det
    logb = (sy - k * sx) / n
    sse = sum((logb + k * xs[i] - ys[i]) ** 2 for i in range(n))
    return sse, k, logb


def grid_search(rows, w1_lo, w1_hi, w1_n, w2_lo, w2_hi, w2_n):
    best = None
    for iw1 in range(w1_n):
        w1 = w1_lo + (w1_hi - w1_lo) * iw1 / max(1, w1_n - 1)
        for iw2 in range(w2_n):
            w2 = w2_lo + (w2_hi - w2_lo) * iw2 / max(1, w2_n - 1)
            res = loglog_fit(rows, w1, w2)
            if res is None:
                continue
            sse, k, logb = res
            if best is None or sse < best[0]:
                best = (sse, k, logb, w1, w2)
    return best


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    rows = []
    idx = 1
    for _ in range(n):
        x0 = float(toks[idx]); x1 = float(toks[idx + 1])
        x2 = float(toks[idx + 2]); y = float(toks[idx + 3])
        idx += 4
        rows.append((x0, x1, x2, y))

    # coarse pass over the coupling-weight plane
    best = grid_search(rows, 0.0, 1.5, 16, 0.0, 1.5, 16)
    if best is None:
        print("1.0*x0")
        return
    _, k, logb, w1, w2 = best

    # local refinement pass around the coarse optimum
    dw = 0.15
    best2 = grid_search(rows, max(0.0, w1 - dw), w1 + dw, 13,
                         max(0.0, w2 - dw), w2 + dw, 13)
    if best2 is not None and best2[0] < best[0]:
        best = best2
    _, k, logb, w1, w2 = best

    b = math.exp(logb)
    print("%.10g * (x0 + %.10g*x1 + %.10g*x2)**%.10g" % (b, w1, w2, k))


if __name__ == "__main__":
    main()
