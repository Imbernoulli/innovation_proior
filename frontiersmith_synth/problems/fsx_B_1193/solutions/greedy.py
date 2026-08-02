# TIER: greedy
# The obvious recipe: this LOOKS like ordinary symbolic regression, so fit a
# memoryless curve y = f(x) (least-squares cubic in x) and ignore the branch
# variable b entirely. It captures the centerline trend of the visible loop
# reasonably well, but the SAME x is visited by both the loading and
# unloading branch at different y -- no function of x alone can separate
# them, so a floor error (~ the average branch gap) survives no matter how
# well the cubic is fit, and it gets worse on the more agitated held-out
# path.
import sys


def solve_linear(A, y):
    """Solve the normal equations (A^T A) c = A^T y by Gaussian elimination
    with partial pivoting. A: list of rows (features), y: list of targets."""
    m = len(A[0])
    ata = [[0.0] * m for _ in range(m)]
    aty = [0.0] * m
    for row, yv in zip(A, y):
        for i in range(m):
            aty[i] += row[i] * yv
            for j in range(m):
                ata[i][j] += row[i] * row[j]
    # ridge regularisation for numerical stability
    for i in range(m):
        ata[i][i] += 1e-9
    n = m
    aug = [ata[i][:] + [aty[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            continue
        aug[col], aug[piv] = aug[piv], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                if factor != 0.0:
                    aug[r] = [av - factor * cv for av, cv in zip(aug[r], aug[col])]
    return [aug[i][n] for i in range(n)]


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("0.0")
        return
    n = int(data[0])
    vals = data[2:]
    xs = [float(vals[2 * i]) for i in range(n)]
    ys = [float(vals[2 * i + 1]) for i in range(n)]

    A = [[1.0, x, x * x, x * x * x] for x in xs]
    c = solve_linear(A, ys)
    k0, k1, k2, k3 = c

    print("%.6f + %.6f * x + %.6f * x ** 2 + %.6f * x ** 3" % (k0, k1, k2, k3))


if __name__ == "__main__":
    main()
