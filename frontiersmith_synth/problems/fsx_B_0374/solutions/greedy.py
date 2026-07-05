# TIER: greedy
"""Log-linear power law (NO floor): fit log L ~ c0 + c1 log C + c2 log R.
Captures the cheap-regime slope but predicts L -> 0 as C,R grow, so it
extrapolates poorly into the large-compute held-out region."""
import sys, math


def main():
    data = sys.stdin.read().split("\n")
    hdr = data[0].split()
    n = int(hdr[0])
    X = []  # [1, logC, logR]
    Y = []
    for i in range(1, n + 1):
        C, R, L = (float(v) for v in data[i].split())
        if L <= 0:
            L = 1e-6
        X.append((1.0, math.log(C), math.log(R)))
        Y.append(math.log(L))

    # normal equations for 3-parameter least squares
    ata = [[0.0] * 3 for _ in range(3)]
    atb = [0.0] * 3
    for row, y in zip(X, Y):
        for a in range(3):
            atb[a] += row[a] * y
            for b in range(3):
                ata[a][b] += row[a] * row[b]

    # solve 3x3 via Gaussian elimination
    M = [ata[r][:] + [atb[r]] for r in range(3)]
    for c in range(3):
        piv = max(range(c, 3), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c] if abs(M[c][c]) > 1e-12 else 1e-12
        for j in range(c, 4):
            M[c][j] /= pv
        for r in range(3):
            if r != c:
                f = M[r][c]
                for j in range(c, 4):
                    M[r][j] -= f * M[c][j]
    c0, c1, c2 = M[0][3], M[1][3], M[2][3]
    A = math.exp(c0)
    # emit floats as standalone whitespace-separated tokens
    sys.stdout.write("%r * x1 ** (%r) * x2 ** (%r)\n" % (A, c1, c2))


if __name__ == "__main__":
    main()
