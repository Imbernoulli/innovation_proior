# TIER: strong
"""
Residual archaeology: fit the dominant mechanism from where it truly
dominates (the LOW-Re half of the sweep), subtract it off every row, then
read the hidden second mechanism's exponent+coefficient from the log-log
slope of the RESIDUAL restricted to the training TAIL (the largest-Re rows,
where the residual signal is cleanest relative to noise). This carries the
correct asymptotic decay rate outside the flume's reachable window, unlike a
single blended power-law fit.
"""
import sys, math


def ols(xs, ys):
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        slope = 0.0
    else:
        slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return slope, intercept


def main():
    data = sys.stdin.read().split()
    idx = 0
    t = int(data[idx]); idx += 1
    n = int(data[idx]); idx += 1
    Res, Cds = [], []
    for _ in range(n):
        Re = float(data[idx]); idx += 1
        Cd = float(data[idx]); idx += 1
        Res.append(Re); Cds.append(Cd)

    # rows are already sorted by Re (gen.py emits them sorted)
    order = sorted(range(n), key=lambda i: Res[i])
    Res = [Res[i] for i in order]
    Cds = [Cds[i] for i in order]

    # Stage 1: fit the dominant power law from the LOW-Re half only, where
    # the subdominant mechanism's contribution is smallest relative to it.
    k_lo = max(8, n // 2)
    xs_lo = [math.log(Res[i]) for i in range(k_lo)]
    ys_lo = [math.log(Cds[i]) for i in range(k_lo)]
    p, log_A = ols(xs_lo, ys_lo)
    A = math.exp(log_A)

    # Stage 2: residual after removing the dominant term, examined on the
    # training TAIL (largest-Re rows), where the residual is largest
    # relative to the noise floor and least contaminated by the dominant
    # term's own small mis-fit near the low end.
    k_tail0 = n - max(20, n // 3)
    xs_res, ys_res = [], []
    for i in range(k_tail0, n):
        resid = Cds[i] - A * (Res[i] ** p)
        if resid > 0.0:
            xs_res.append(math.log(Res[i]))
            ys_res.append(math.log(resid))

    if len(xs_res) >= 3:
        q, log_B = ols(xs_res, ys_res)
        B = math.exp(log_B)
        if not (q > p and B > 0.0):
            print("%.10g * powv(Re, %.10g)" % (A, p))
            return
        # A shallow subdominant mechanism decays slower than the dominant
        # one but is still a genuine DRAG law -- it cannot grow with Re.
        # A near-zero training-window signal can make the raw tail-slope
        # estimate noisy or even mis-signed; clip to the physically sane
        # "still decaying, still subdominant at the training tail" band
        # rather than trusting an unbounded extrapolated slope.
        q = max(-0.60, min(-0.01, q))
        B = max(1e-6, min(20.0 * A, B))
        print("%.10g * powv(Re, %.10g) + %.10g * powv(Re, %.10g)" % (A, p, B, q))
    else:
        print("%.10g * powv(Re, %.10g)" % (A, p))


if __name__ == "__main__":
    main()
