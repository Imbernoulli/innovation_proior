# TIER: greedy
"""
The obvious "recipe" fit: ordinary least squares of cost on (count_a, count_b,
1) -- a per-symbol additive linear model, completely blind to WHERE in the
string each symbol occurs (so blind to any lane-switch / regime change).
Emitted as a 1-state automaton (self-loop weights = fitted alpha/beta, final
weight = fitted intercept gamma). Fits the training notebook reasonably (one
route dominates most short trips) but fails to track the true cost once the
optimal route has switched on long trips.
"""
import sys


def solve_3x3(A):
    # Gaussian elimination on a 3x4 augmented matrix (in place). A is list of 3 rows of 4 floats.
    for i in range(3):
        piv = A[i][i]
        if abs(piv) < 1e-12:
            piv = 1e-12 if piv >= 0 else -1e-12
        for j in range(i, 4):
            A[i][j] /= piv
        for k in range(3):
            if k != i:
                f = A[k][i]
                for j in range(i, 4):
                    A[k][j] -= f * A[i][j]
    return A[0][3], A[1][3], A[2][3]


def main():
    data = sys.stdin.read().split("\n")
    first = data[0].split()
    n = int(first[1])
    rows = []
    for i in range(2, 2 + n):
        parts = data[i].split()
        s, c = parts[0], int(parts[1])
        rows.append((s, c))

    Sxx = [[0.0] * 3 for _ in range(3)]
    Sxy = [0.0] * 3
    for s, c in rows:
        na = s.count("a")
        nb = s.count("b")
        x = [na, nb, 1.0]
        for i in range(3):
            Sxy[i] += x[i] * c
            for j in range(3):
                Sxx[i][j] += x[i] * x[j]

    A = [Sxx[i] + [Sxy[i]] for i in range(3)]
    alpha, beta, gamma = solve_3x3(A)

    out = []
    out.append("1 2")
    out.append("0 a 0 %.6f" % alpha)
    out.append("0 b 0 %.6f" % beta)
    out.append("0")
    out.append("1")
    out.append("0 %.6f" % gamma)
    print("\n".join(out))


if __name__ == "__main__":
    main()
