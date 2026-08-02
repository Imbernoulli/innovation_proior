# TIER: greedy
# The obvious recipe: flow reacts to TODAY's weather plus a short memory of
# recent rain (a stand-in for "antecedent moisture"). Fit a linear model of
# flow on [1, p, relu(tm), p*step(tm), lag_avg, p*lag_avg] by ordinary least
# squares on the training rows -- no persisting state anywhere. This tracks
# the RAIN-season training data well (flow really is close to a memoryless
# function of recent weather there), but on the held-out winter-into-spring
# season it has nothing that can carry a snowpack across a long cold spell,
# so it collapses to near-baseflow whenever precip is low even while stored
# snow is still melting.
import sys


def solve_ls(X, y, ridge=1e-6):
    m = len(X[0])
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for row, yv in zip(X, y):
        for i in range(m):
            b[i] += row[i] * yv
            for j in range(m):
                A[i][j] += row[i] * row[j]
    for i in range(m):
        A[i][i] += ridge
    # Gaussian elimination with partial pivoting
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-12:
            continue
        A[col], A[piv] = A[piv], A[col]
        b[col], b[piv] = b[piv], b[col]
        pv = A[col][col]
        for j in range(col, m):
            A[col][j] /= pv
        b[col] /= pv
        for r in range(m):
            if r == col:
                continue
            f = A[r][col]
            if f == 0.0:
                continue
            for j in range(col, m):
                A[r][j] -= f * A[col][j]
            b[r] -= f * b[col]
    return b


def main():
    data = sys.stdin.read().split()
    if not data:
        print("OUT 0.3"); return
    n = int(data[0])
    vals = data[2:]
    p = [0.0] * n
    tm = [0.0] * n
    y = [0.0] * n
    for i in range(n):
        p[i] = float(vals[4 * i])
        tm[i] = float(vals[4 * i + 1])
        y[i] = float(vals[4 * i + 2])

    W = 6
    X = []
    for t in range(n):
        lag_avg = sum(p[t - j] if t - j >= 0 else 0.0 for j in range(1, W + 1)) / W
        warm = tm[t] if tm[t] > 0.0 else 0.0
        stepwarm = 1.0 if tm[t] > 0.0 else 0.0
        X.append([1.0, p[t], warm, p[t] * stepwarm, lag_avg, p[t] * lag_avg])
    w = solve_ls(X, y)

    print("OUT %.6f + %.6f * p + %.6f * relu ( tm ) + %.6f * ( p * step ( tm ) ) "
          "+ %.6f * ( ( pk1 + pk2 + pk3 + pk4 + pk5 + pk6 ) / 6.0 ) "
          "+ %.6f * ( p * ( ( pk1 + pk2 + pk3 + pk4 + pk5 + pk6 ) / 6.0 ) )"
          % (w[0], w[1], w[2], w[3], w[4], w[5]))


if __name__ == "__main__":
    main()
