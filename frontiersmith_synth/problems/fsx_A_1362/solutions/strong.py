# TIER: strong
"""The insight: envy-freeness is not something to check *after* optimizing -- it
is a linear certificate condition (one inequality per ordered agent pair) that
can be baked directly into the feasible region of a linear program. Formulate

    maximize   sum_i sum_j v[i][j] * x[i][j]
    subject to sum_i x[i][j] = 1                       for every item j
               sum_j v[i][j]*x[k][j] <= sum_j v[i][j]*x[i][j]   for every i != k
               0 <= x[i][j] <= 1

and solve it directly. Because items are FRACTIONALLY divisible, the LP can
split a contested item across the agents who prize it instead of handing it
wholly to a single winner (which is all the whole-item greedy can ever do),
recovering welfare the whole-item approach structurally cannot reach while
still certifying zero envy by construction."""
import sys

from scipy.optimize import linprog


def solve(n, m, v):
    nv = n * m

    def idx(i, j):
        return i * m + j

    c = [0.0] * nv
    for i in range(n):
        for j in range(m):
            c[idx(i, j)] = -float(v[i][j])  # linprog minimizes -> negate to maximize

    A_eq = []
    b_eq = []
    for j in range(m):
        row = [0.0] * nv
        for i in range(n):
            row[idx(i, j)] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)

    A_ub = []
    b_ub = []
    for i in range(n):
        for k in range(n):
            if k == i:
                continue
            row = [0.0] * nv
            for j in range(m):
                row[idx(k, j)] += float(v[i][j])
                row[idx(i, j)] -= float(v[i][j])
            A_ub.append(row)
            b_ub.append(0.0)

    bounds = [(0.0, 1.0)] * nv
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if res.success:
        x_flat = list(res.x)
    else:
        # LP is always feasible (equal split satisfies every constraint), so this
        # should never trigger; fall back to equal split defensively.
        x_flat = [1.0 / n] * nv

    x = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            val = x_flat[idx(i, j)]
            x[i][j] = val if val > 1e-12 else 0.0

    # clean up tiny numerical drift so every column sums to exactly 1
    for j in range(m):
        s = sum(x[i][j] for i in range(n))
        if s > 1e-9:
            for i in range(n):
                x[i][j] /= s
        else:
            for i in range(n):
                x[i][j] = 1.0 / n
    return x


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    v = [[int(next(it)) for _ in range(m)] for _ in range(n)]

    x = solve(n, m, v)

    out_lines = []
    for j in range(m):
        out_lines.append(" ".join(f"{x[i][j]:.9f}" for i in range(n)))
    sys.stdout.write("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main()
