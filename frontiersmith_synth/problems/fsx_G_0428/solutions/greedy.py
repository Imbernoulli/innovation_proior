# TIER: greedy
# Fit a degree-3 polynomial in the index n (centred/scaled for conditioning) by
# ordinary least squares on the observed prefix, then emit it as a closed form
# in n.  Excellent on figurate (polynomial) sequences, but a polynomial cannot
# track exponential n-nacci growth, so it extrapolates poorly there.
import sys


def solve(M):
    n = len(M)
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col] or 1e-12
        for k in range(col, n + 1):
            M[col][k] /= pv
        for r in range(n):
            if r != col:
                f = M[r][col]
                for k in range(col, n + 1):
                    M[r][k] -= f * M[col][k]
    return [M[j][n] for j in range(n)]


def main():
    data = sys.stdin.read().split("\n")
    T = int(data[0].split()[0])
    ns = []; ys = []
    for ln in data[1:1 + T]:
        p = ln.split()
        if len(p) >= 2:
            ns.append(float(p[0])); ys.append(float(p[1]))
    nmid = (T - 1) / 2.0
    nscale = max(1.0, (T - 1) / 2.0)
    deg = 3
    # normal equations for basis {1, u, u^2, u^3}, u=(n-nmid)/nscale
    A = [[0.0] * (deg + 2) for _ in range(deg + 1)]
    for nn, yy in zip(ns, ys):
        u = (nn - nmid) / nscale
        powu = [u ** k for k in range(deg + 1)]
        for i in range(deg + 1):
            A[i][deg + 1] += powu[i] * yy
            for k in range(deg + 1):
                A[i][k] += powu[i] * powu[k]
    b = solve(A)
    u = "((n - %r) / %r)" % (nmid, nscale)
    terms = ["%r" % b[0]]
    terms.append("%r * %s" % (b[1], u))
    terms.append("%r * %s**2" % (b[2], u))
    terms.append("%r * %s**3" % (b[3], u))
    print(" + ".join(terms))


if __name__ == "__main__":
    main()
