# TIER: strong
# The insight: the output is not a function of x, only of x's HISTORY -- so
# represent the internal branch state explicitly instead of pattern-matching
# "fit y=f(x)". Derive the branch sequence b from the training x's directions
# using the FIXED, stated rule (the same rule the grader applies on the
# held-out path), then fit BOTH the centerline g(x) and a branch-modulated
# gap c(x) via one joint least-squares fit on features
# [1, x, x^2, x^3, b, b*x^2]. Emitting an expression that references b
# explicitly generalises across held-out paths with new reversal points and
# a different sampling rate, because the branch is rate-independent.
import sys


def branch_states(xs):
    b = [1]
    for i in range(1, len(xs)):
        if xs[i] > xs[i - 1]:
            b.append(1)
        elif xs[i] < xs[i - 1]:
            b.append(-1)
        else:
            b.append(b[-1])
    return b


def solve_linear(A, y):
    m = len(A[0])
    ata = [[0.0] * m for _ in range(m)]
    aty = [0.0] * m
    for row, yv in zip(A, y):
        for i in range(m):
            aty[i] += row[i] * yv
            for j in range(m):
                ata[i][j] += row[i] * row[j]
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
    bs = branch_states(xs)

    A = [[1.0, x, x * x, x * x * x, b, b * x * x] for x, b in zip(xs, bs)]
    c = solve_linear(A, ys)
    k0, k1, k2, k3, g0, g1 = c

    print("%.6f + %.6f * x + %.6f * x ** 2 + %.6f * x ** 3 + b * ( %.6f + %.6f * x ** 2 )"
          % (k0, k1, k2, k3, g0, g1))


if __name__ == "__main__":
    main()
