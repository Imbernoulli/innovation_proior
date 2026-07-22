# TIER: greedy
# The obvious recipe for "three numeric predictor columns, one numeric
# target": fit a single global MULTIPLICATIVE power law
#     Q = k0 * D^a * d^b * rho^c
# by ordinary least squares in log-log space over all three predictors at
# once. This recovers rho's exponent essentially correctly (rho is sampled
# independently of D and d, so nothing confounds it) and tracks the
# small-hopper training band's scale reasonably -- a practitioner who checks
# the training residual sees a good fit and stops here. But a multiplicative
# D^a * d^b shape can NEVER express an additive offset between D and d: it
# has no way to represent "the flow only sees D minus a chunk proportional
# to d". Near the training band's small apertures that offset is a
# significant fraction of D, so this fit partially (and wrongly) absorbs the
# offset's curvature into a biased exponent `a`, plus an accidental,
# wrong-sign dependence `b` on d. On the held-out grid -- apertures 2-20x
# wider, grain sizes never seen in training -- the same absolute offset is a
# much smaller fraction of D, so the biased exponents predict the wrong
# growth rate and drift increasingly off.
import sys, math


def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        dv = M[c][c]
        if abs(dv) < 1e-18:
            dv = 1e-18
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / dv
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][n] / (M[i][i] if abs(M[i][i]) > 1e-18 else 1e-18) for i in range(n)]


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("1.0"); return
    n = int(data[1])
    vals = data[2:]
    feats, ys = [], []
    for i in range(n):
        D = float(vals[4 * i]); d = float(vals[4 * i + 1])
        rho = float(vals[4 * i + 2]); q = float(vals[4 * i + 3])
        feats.append([1.0, math.log(D), math.log(d), math.log(rho)])
        ys.append(math.log(q))
    m = 4
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for x, y in zip(feats, ys):
        for r in range(m):
            b[r] += x[r] * y
            for c in range(m):
                A[r][c] += x[r] * x[c]
    logk0, a, bexp, c = solve(A, b)
    k0 = math.exp(logk0)
    print("%.10g * powv(D, %.10g) * powv(d, %.10g) * powv(rho, %.10g)"
          % (k0, a, bexp, c))


if __name__ == "__main__":
    main()
