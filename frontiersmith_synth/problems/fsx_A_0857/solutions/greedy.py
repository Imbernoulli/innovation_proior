# TIER: greedy
# Textbook multi-variable recipe: ordinary least-squares LINEAR regression
# y = c0 + c1*x0 + c2*x1 + c3*x2 on all three flows. This is the "obvious"
# approach for tabular multi-variable data -- it DOES notice the feeders
# matter (unlike an x0-only fit) but it assumes an ADDITIVE LINEAR effect.
# Over the light-traffic training box a linear model can locally track the
# power-law curve well enough to look plausible, but it never recovers the
# SHARED SUPERLINEAR EXPONENT, so it drastically under-predicts once the
# effective combined load is pushed into the heavy-traffic regime.
import sys


def solve_linear(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[piv] = M[piv], M[i]
        if abs(M[i][i]) < 1e-12:
            continue
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            for c in range(i, n + 1):
                M[r][c] -= f * M[i][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0
    return x


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    rows = []
    idx = 1
    for _ in range(n):
        x0 = float(toks[idx]); x1 = float(toks[idx + 1])
        x2 = float(toks[idx + 2]); y = float(toks[idx + 3])
        idx += 4
        rows.append((x0, x1, x2, y))

    A = [[0.0] * 4 for _ in range(4)]
    bvec = [0.0] * 4
    for x0, x1, x2, y in rows:
        feats = [1.0, x0, x1, x2]
        for i in range(4):
            bvec[i] += feats[i] * y
            for j in range(4):
                A[i][j] += feats[i] * feats[j]
    c0, c1, c2, c3 = solve_linear(A, bvec)

    print("%.10g + %.10g*x0 + %.10g*x1 + %.10g*x2" % (c0, c1, c2, c3))


if __name__ == "__main__":
    main()
