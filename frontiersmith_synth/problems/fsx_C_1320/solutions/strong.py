# TIER: strong
# The insight: a memoryless "band gap vs composition" curve averages away the
# dopant identity, but the residuals of that curve correlate with the
# dopant's ELECTRONEGATIVITY mismatch (nonlinearly -- small mismatch barely
# matters, large mismatch dominates: a different mechanism) and its RADIUS
# mismatch.  Rather than adding flexible x-only terms (which cannot see
# chemistry at all), fit a physically-grounded model that ALSO includes
# x*dEN^2 (the nonlinear electronegativity channel) and x*dR (the radius
# mismatch channel).  Because these channels have small variance inside the
# narrow visible family, fit them in STANDARDIZED feature space with a small
# ridge penalty so the estimate stays stable instead of exploding on the
# nearly-collinear low-variance columns -- then emit the closed-form
# predictor.  This is the term that lets the model extrapolate to dopants far
# outside the training chemistry, where the greedy curve has no mechanism at
# all.
import sys


def solve_ls(X, y, ridge=0.0):
    n = len(X)
    m = len(X[0])
    XtX = [[0.0] * m for _ in range(m)]
    Xty = [0.0] * m
    for row, yy in zip(X, y):
        for i in range(m):
            Xty[i] += row[i] * yy
            for j in range(m):
                XtX[i][j] += row[i] * row[j]
    if ridge > 0:
        for i in range(1, m):  # never regularize the intercept (index 0)
            XtX[i][i] += ridge
    M = [XtX[i][:] + [Xty[i]] for i in range(m)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12:
            continue
        for r in range(m):
            if r != col:
                f = M[r][col] / M[col][col]
                for c in range(col, m + 1):
                    M[r][c] -= f * M[col][c]
    return [M[i][m] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(m)]


def main():
    data = sys.stdin.read().split()
    n = int(data[1])
    xs, dens, drs, ys = [], [], [], []
    for i in range(n):
        base = 2 + 5 * i
        xs.append(float(data[base + 1]))
        dens.append(float(data[base + 2]))
        drs.append(float(data[base + 3]))
        ys.append(float(data[base + 4]))

    # physically-grounded feature set: composition (x, x^2) + electronegativity
    # NONLINEARITY (x*dEN^2) + radius mismatch (x*dR)
    feats = [[x, x * x, x * d * d, x * r] for x, d, r in zip(xs, dens, drs)]
    m = 4

    means = [sum(f[i] for f in feats) / n for i in range(m)]
    stds = []
    for i in range(m):
        var = sum((f[i] - means[i]) ** 2 for f in feats) / n
        stds.append(var ** 0.5 if var > 1e-12 else 1.0)

    Z = [[1.0] + [(f[i] - means[i]) / stds[i] for i in range(m)] for f in feats]
    coef = solve_ls(Z, ys, ridge=2.0)

    # fold the standardization back out so we can emit a PLAIN closed-form
    # expression in the original (x, dEN, dR) variables
    orig = [coef[i + 1] / stds[i] for i in range(m)]
    intercept = coef[0] - sum(coef[i + 1] * means[i] / stds[i] for i in range(m))

    print("(%.6f) + (%.6f) * x + (%.6f) * x ** 2 + (%.6f) * x * dEN ** 2 + (%.6f) * x * dR"
          % (intercept, orig[0], orig[1], orig[2], orig[3]))


if __name__ == "__main__":
    main()
