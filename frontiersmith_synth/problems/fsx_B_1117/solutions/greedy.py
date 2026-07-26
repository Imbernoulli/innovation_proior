# TIER: greedy
# The obvious recipe: assume the reaction is FIRST ORDER (a plain bilinear
# response surface) and least-squares fit
#     rate ~= d + a*S + b*C + c*S*C
# on the dilute training rows. Because the true saturating law is locally
# linear for S << Km, this fits the training data almost perfectly -- but
# it has no saturation built in, so its slope keeps extrapolating forever.
# It overshoots badly once S runs past the training envelope into the
# regime where the true reaction levels off.
import sys


def solve4(XtX, Xty):
    A = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]
    m = 4
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        if abs(A[col][col]) < 1e-12:
            continue
        for r in range(m):
            if r != col:
                factor = A[r][col] / A[col][col]
                for c in range(col, m + 1):
                    A[r][c] -= factor * A[col][c]
    return [A[i][m] / A[i][i] if abs(A[i][i]) > 1e-12 else 0.0 for i in range(m)]


def main():
    data = sys.stdin.read().split()
    if len(data) < 3:
        print("1.0")
        return
    n_regimes = int(data[1])
    n_pts = int(data[2])
    vals = data[3:]
    n = n_regimes * n_pts
    rows = []
    for i in range(n):
        S = float(vals[3 * i])
        C = float(vals[3 * i + 1])
        r = float(vals[3 * i + 2])
        rows.append((S, C, r))

    XtX = [[0.0] * 4 for _ in range(4)]
    Xty = [0.0] * 4
    for S, C, r in rows:
        x = [1.0, S, C, S * C]
        for a in range(4):
            Xty[a] += x[a] * r
            for b in range(4):
                XtX[a][b] += x[a] * x[b]
    d, a, b, c = solve4(XtX, Xty)

    print("%.8f + %.8f * S + %.8f * C + %.8f * S * C" % (d, a, b, c))


if __name__ == "__main__":
    main()
