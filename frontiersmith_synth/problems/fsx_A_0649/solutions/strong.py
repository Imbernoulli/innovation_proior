# TIER: strong
"""The insight: this is NOT one smooth function -- it is a hybrid automaton.
Once you suspect two regimes, (1) find the natural gap in the displacement
values to get an initial stuck/slip split, (2) alternately refit each mode's
TRIVIALLY SIMPLE law (stuck: through-origin creep in F*ln(1+r); slip: linear
in F) and reclassify every point to whichever mode fits it better (hard-EM),
(3) once the split stabilises, fit a linear separator in (F, ln(1+r)) space
to recover the aging guard itself. Each sub-step is a two-line least squares
fit; the "hard" part is realising the decomposition exists at all."""
import sys, math


def solve_lin(A, b):
    """Solve (A^T A) x = A^T b via Gaussian elimination w/ partial pivoting."""
    n = len(A[0])
    ATA = [[sum(A[k][i] * A[k][j] for k in range(len(A))) for j in range(n)] for i in range(n)]
    ATb = [sum(A[k][i] * b[k] for k in range(len(A))) for i in range(n)]
    M = [row[:] + [ATb[i]] for i, row in enumerate(ATA)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            continue
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [x / pv for x in M[col]]
        for r in range(n):
            if r != col:
                f = M[r][col]
                M[r] = [M[r][k] - f * M[col][k] for k in range(n + 1)]
    return [M[i][n] for i in range(n)]


def fit_stuck(pts):
    """y = kc * x, x = F*L, through the origin (closed form)."""
    num = sum(x * y for x, y in pts)
    den = sum(x * x for x, y in pts)
    return num / den if den > 1e-9 else 0.03


def fit_slip(pts):
    """y = a*F + b (2-param OLS)."""
    if len(pts) < 2:
        return 0.8, 0.2
    A = [[1.0, F] for F, y in pts]
    b = [y for F, y in pts]
    b0, a0 = solve_lin(A, b)
    return a0, b0


def main():
    data = sys.stdin.read().split()
    n_train = int(data[0])
    idx = 2
    rows = []
    for _ in range(n_train):
        F = float(data[idx]); r = float(data[idx + 1]); y = float(data[idx + 2])
        idx += 3
        L = math.log(1.0 + r)
        rows.append((F, L, y))

    # --- initial split: largest gap in sorted y (natural two-cluster break) ---
    order = sorted(range(len(rows)), key=lambda i: rows[i][2])
    ys_sorted = [rows[i][2] for i in order]
    if len(ys_sorted) >= 4:
        gaps = [(ys_sorted[k + 1] - ys_sorted[k], k) for k in range(len(ys_sorted) - 1)]
        lo = max(1, len(ys_sorted) // 10)
        hi = len(ys_sorted) - 1 - lo
        cand = [g for g in gaps if lo <= g[1] <= hi] or gaps
        _, kbest = max(cand)
        cut = ys_sorted[kbest]
    else:
        cut = sum(ys_sorted) / len(ys_sorted) if ys_sorted else 0.0
    labels = [1 if rows[i][2] > cut else 0 for i in range(len(rows))]

    kc, aslip, bslip = 0.03, 0.8, 0.2
    for _ in range(25):
        stuck_pts = [(F * L, y) for (F, L, y), lab in zip(rows, labels) if lab == 0]
        slip_pts = [(F, y) for (F, L, y), lab in zip(rows, labels) if lab == 1]
        if stuck_pts:
            kc = fit_stuck(stuck_pts)
        if slip_pts:
            aslip, bslip = fit_slip(slip_pts)
        new_labels = []
        for F, L, y in rows:
            pred_stuck = kc * F * L
            pred_slip = aslip * F + bslip
            new_labels.append(0 if abs(y - pred_stuck) <= abs(y - pred_slip) else 1)
        if new_labels == labels:
            labels = new_labels
            break
        labels = new_labels

    # --- recover the guard: a REGULARISED LOGISTIC classifier of label on
    # (F, L), fit by gradient descent on standardised features. This pools
    # evidence from every point (robust to a couple of near-boundary label
    # flips from noise), unlike order statistics that key off single points.
    f0_hat, a_hat = 3.0, 1.0
    if any(l == 0 for l in labels) and any(l == 1 for l in labels):
        Fs = [F for F, L, y in rows]
        Ls = [L for F, L, y in rows]
        mF = sum(Fs) / len(Fs)
        sF = (sum((f - mF) ** 2 for f in Fs) / len(Fs)) ** 0.5 or 1.0
        mL = sum(Ls) / len(Ls)
        sL = (sum((l - mL) ** 2 for l in Ls) / len(Ls)) ** 0.5 or 1.0
        Xs = [((F - mF) / sF, (L - mL) / sL) for F, L, y in rows]
        w0 = w1 = w2 = 0.0
        lr, l2, n_iter = 0.3, 0.02, 800
        m = len(rows)
        for _ in range(n_iter):
            g0 = g1 = g2 = 0.0
            for (xf, xl), lab in zip(Xs, labels):
                z = max(-30.0, min(30.0, w0 + w1 * xf + w2 * xl))
                p = 1.0 / (1.0 + math.exp(-z))
                err = p - lab
                g0 += err; g1 += err * xf; g2 += err * xl
            g0 /= m; g1 /= m; g2 /= m
            w0 -= lr * g0
            w1 -= lr * (g1 + l2 * w1)
            w2 -= lr * (g2 + l2 * w2)
        if abs(w1) > 1e-9:
            f0_hat = mF - sF * w0 / w1 + sF * w2 * mL / (sL * w1)
            a_hat = -sF * w2 / (sL * w1)

    expr = ("( ( %.8f ) * F * log ( 1 + r ) ) if ( F <= ( ( %.8f ) + ( %.8f ) * log ( 1 + r ) ) ) "
            "else ( ( %.8f ) * F + ( %.8f ) )") % (kc, f0_hat, a_hat, aslip, bslip)
    print(expr)


if __name__ == "__main__":
    main()
