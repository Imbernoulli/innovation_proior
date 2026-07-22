# TIER: strong
# The insight: no single curve in x=r/t explains the training cloud once you
# notice a MINORITY of training rows (thin sheets with unusually large r/t)
# already sit on a visibly different branch than the rest. That is the
# signature of a second regime, not noise. So posit a regime boundary
#   xc(t) = A - Bc * t
# and grid-search (A, Bc): for each candidate, split the training rows by
# x=r/t vs xc(t), fit an ELASTIC branch (through-origin least squares in x)
# on rows below the boundary and a PLASTIC branch (least squares in x^(1/3))
# on rows above it, and keep the boundary that minimises total training MSE.
# Emitting the branch-selecting expression via step(...) gating lets the
# SAME formula reclassify held-out rows correctly even though the boundary
# has moved into a completely different part of (r,t) space (thick sheets),
# because the boundary -- not just the branch shapes -- was recovered.
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        print("1.2"); return
    n = int(data[0])
    vals = data[2:]
    rows = []
    for i in range(n):
        r = float(vals[3 * i])
        t = float(vals[3 * i + 1])
        s = float(vals[3 * i + 2])
        x = r / t
        rows.append((r, t, x, s))

    RIDGE = 3.0  # mild L2 shrinkage: keeps small/noisy buckets from swinging wildly

    def fit_elastic(pts):
        # through-origin ridge least squares: S = c1 * x
        sxx = sum(x * x for (_, _, x, _) in pts)
        sxs = sum(x * s for (_, _, x, s) in pts)
        return sxs / (sxx + RIDGE) if sxx > 1e-9 else 0.05

    def fit_plastic(pts):
        # ridge least squares in u = x^(1/3): S = c2*u + c3 (centred so the
        # ridge penalty only shrinks the SLOPE, not the fitted mean level)
        m = len(pts)
        us = [x ** (1.0 / 3.0) for (_, _, x, _) in pts]
        ss = [s for (_, _, _, s) in pts]
        mu = sum(us) / m
        ms = sum(ss) / m
        suu = sum((u - mu) ** 2 for u in us)
        sus = sum((u - mu) * (sv - ms) for u, sv in zip(us, ss))
        c2 = sus / (suu + RIDGE) if suu > 1e-9 else 1.0
        c3 = ms - c2 * mu
        return c2, c3

    # 3-fold cross-validation over the boundary grid: pick the (A,Bc) whose
    # branch fits generalise BEST across folds (not just the one that lets a
    # 2-parameter branch overfit a handful of noisy points on all the data).
    K = 3
    folds = [rows[k::K] for k in range(K)]
    min_pts = max(10, n // 10)

    def split(pts, A0, B0):
        below, above = [], []
        for (r, t, x, s) in pts:
            xc = A0 - B0 * t
            if xc < 0.5:
                xc = 0.5
            (below if x < xc else above).append((r, t, x, s))
        return below, above

    best = None
    best_AB = (30.0, 11.0)
    A_grid = [18.0 + 1.0 * i for i in range(19)]     # 18..36
    B_grid = [3.0 + 1.0 * i for i in range(15)]       # 3..17
    for A0 in A_grid:
        for B0 in B_grid:
            below_all, above_all = split(rows, A0, B0)
            if len(below_all) < min_pts or len(above_all) < min_pts:
                continue
            cv_se = 0.0
            cv_n = 0
            ok = True
            for k in range(K):
                fit_rows = [row for j, fold in enumerate(folds) if j != k for row in fold]
                val_rows = folds[k]
                fbelow, fabove = split(fit_rows, A0, B0)
                if len(fbelow) < min_pts // 2 or len(fabove) < min_pts // 2:
                    ok = False
                    break
                c1k = fit_elastic(fbelow)
                c2k, c3k = fit_plastic(fabove)
                for (r, t, x, s) in val_rows:
                    xc = max(0.5, A0 - B0 * t)
                    pred = c1k * x if x < xc else c2k * (x ** (1.0 / 3.0)) + c3k
                    cv_se += (pred - s) ** 2
                    cv_n += 1
            if not ok or cv_n == 0:
                continue
            mse = cv_se / cv_n
            if best is None or mse < best:
                best = mse
                best_AB = (A0, B0)

    # Refit the final branch coefficients on ALL rows using the boundary
    # chosen by cross-validation, so no data is wasted in the reported model.
    A0, B0 = best_AB
    below, above = split(rows, A0, B0)
    if len(below) < 2:
        below = rows
    if len(above) < 2:
        above = rows
    c1 = fit_elastic(below)
    c2, c3 = fit_plastic(above)
    expr = (
        "( %.6f * ( r / t ) ) * step ( ( %.6f - %.6f * t ) - ( r / t ) ) "
        "+ ( %.6f * ( r / t ) ** ( 1.0 / 3.0 ) + %.6f ) * "
        "step ( ( r / t ) - ( %.6f - %.6f * t ) )"
        % (c1, A0, B0, c2, c3, A0, B0)
    )
    print(expr)


if __name__ == "__main__":
    main()
