# TIER: greedy
# The obvious recipe: treat this as plain curve fitting.  Regress a smooth
# quadratic T ~ c0 + c1*Ta + c2*Ta^2 on the sub-critical training rows by
# ordinary least squares and just keep evaluating it everywhere -- never
# switch to a different branch (THRESH set to a sentinel far beyond any
# query so BELOW always fires).  This tracks the visible near-critical
# curvature reasonably (the pre-threshold branch really is smooth), but it
# has no notion that the cooling hardware has a hard capacity limit: past
# the last training ambient it just keeps extrapolating the same smooth
# climb, missing the super-critical jump to Tfail by a wide margin, and it
# ignores the given b/h/Hmax hints entirely.
import sys


def solve(M_in, rhs_in):
    n = len(M_in)
    M = [row[:] + [rhs_in[i]] for i, row in enumerate(M_in)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        d = M[c][c]
        if abs(d) < 1e-18:
            d = 1e-18
        for r in range(n):
            if r == c:
                continue
            f = M[r][c] / d
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    return [M[i][n] / (M[i][i] if abs(M[i][i]) > 1e-18 else 1e-18) for i in range(n)]


def main():
    data = sys.stdin.read().split()
    if not data:
        print("THRESH 1e9"); print("BELOW 0.0"); print("ABOVE 0.0"); return
    n = int(data[0])
    rest = data[2:]
    rows_tok = rest[4:]  # skip b h Hmax Tfail
    Tas = [float(rows_tok[2 * i]) for i in range(n)]
    Ts = [float(rows_tok[2 * i + 1]) for i in range(n)]

    feats = [[1.0, x, x * x] for x in Tas]
    m = 3
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for x, y in zip(feats, Ts):
        for r in range(m):
            b[r] += x[r] * y
            for c in range(m):
                A[r][c] += x[r] * x[c]
    c0, c1, c2 = solve(A, b)

    print("THRESH 1e9")
    print("BELOW %.10g + %.10g*Ta + %.10g*Ta**2" % (c0, c1, c2))
    print("ABOVE %.10g + %.10g*Ta + %.10g*Ta**2" % (c0, c1, c2))


if __name__ == "__main__":
    main()
