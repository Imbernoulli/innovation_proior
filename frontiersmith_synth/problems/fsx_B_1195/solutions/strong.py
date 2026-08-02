# TIER: strong
# The insight: the visible trace is a SUM of an irreversible settlement curve
# D0 + D1*(1-exp(-t/tau)) + D2*t and a reversible term driven directly by the
# TEMPERATURE reading, D3*T. Decompose first, using T as the regressor that
# disentangles the two, THEN extrapolate only the (t-only) irreversible piece
# forward while reading the reversible piece off whatever temperature the
# grader supplies at each held-out point.
#
# A naive full-training least-squares fit of (D0,D1,tau,D2,D3) is numerically
# degenerate when tau is comparable to the visible span: (1-exp(-t/tau)) and
# t become nearly collinear, and training RSS alone cannot tell "near-done
# settling" from "still slowly climbing" apart. Instead: hold out the LAST
# 20% of the visible span as a validation slice, grid-search tau by how well
# each candidate law -- fit with a small ridge penalty for stability --
# predicts that held-back slice (a genuine within-visible extrapolation
# test), then refit on all visible data at the winning tau.
import sys, math


def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        if abs(d) < 1e-18:
            d = 1e-18
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / d
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][n] / (M[i][i] if abs(M[i][i]) > 1e-18 else 1e-18) for i in range(n)]


RIDGE = 0.006


def fit4(rows, tau, ridge=0.0):
    feats = []
    for tt, Traw, d in rows:
        feats.append([1.0, 1.0 - math.exp(-tt / tau), tt, Traw])
    m = 4
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for x, (tt, Traw, d) in zip(feats, rows):
        for r in range(m):
            b[r] += x[r] * d
            for c in range(m):
                A[r][c] += x[r] * x[c]
    if ridge:
        n = len(rows)
        for i in range(1, m):
            A[i][i] += ridge * n
    return solve(A, b)


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    n = int(data[0])
    vals = data[2:]
    rows = []
    for i in range(n):
        tt = float(vals[3 * i])
        Traw = float(vals[3 * i + 1])
        d = float(vals[3 * i + 2])
        rows.append((tt, Traw, d))

    tmax = max(r[0] for r in rows)
    cut = tmax * 0.8
    fit_rows = [r for r in rows if r[0] <= cut]
    val_rows = [r for r in rows if r[0] > cut]
    if len(fit_rows) < 5 or len(val_rows) < 3:
        fit_rows, val_rows = rows, rows

    best = None
    tau = 40.0
    for _ in range(30):
        coef = fit4(fit_rows, tau, ridge=RIDGE)
        vrss = 0.0
        for tt, Traw, d in val_rows:
            pred = coef[0] + coef[1] * (1.0 - math.exp(-tt / tau)) + coef[2] * tt + coef[3] * Traw
            vrss += (pred - d) ** 2
        if best is None or vrss < best[0] - 1e-9:
            best = (vrss, tau)
        tau *= 1.12

    tau_best = best[1]
    D0, D1, D2, D3 = fit4(rows, tau_best, ridge=RIDGE)

    print("%.10g + %.10g * (1 - exp(-t / %.10g)) + %.10g * t + %.10g * T"
          % (D0, D1, tau_best, D2, D3))


if __name__ == "__main__":
    main()
