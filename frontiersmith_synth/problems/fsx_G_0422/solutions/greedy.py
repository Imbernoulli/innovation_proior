# TIER: greedy
# Additive model with the right split (floor + strain term + rate term) but the WRONG
# curvature in strain: linear in s instead of the concave power law.  Near s=0 the Ludwik
# curve s^n (n<1) rises very steeply, so a line fitted on the low-strain sample has a large
# slope and OVERSHOOTS when extrapolated to the high-strain held-out coupon.
#   sigma = a + b*s + c*log(r)
# Beats the plain power-law baseline (it captures the additive floor) but loses to the
# strong law (which recovers the true hardening exponent).
import sys, math


def lstsq(rows, y):
    m = len(rows[0])
    A = [[0.0] * m for _ in range(m)]
    bvec = [0.0] * m
    for r, yy in zip(rows, y):
        for i in range(m):
            bvec[i] += r[i] * yy
            for j in range(m):
                A[i][j] += r[i] * r[j]
    M = [A[i][:] + [bvec[i]] for i in range(m)]
    for c in range(m):
        piv = max(range(c, m), key=lambda rr: abs(M[rr][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(m):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, m + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][m] / M[i][i] for i in range(m)]


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0])
    idx = 1
    rows, y = [], []
    for _ in range(n):
        s = float(toks[idx]); r = float(toks[idx + 1]); d = float(toks[idx + 2])
        idx += 3
        rows.append([1.0, s, math.log(r)])
        y.append(d)
    a, b, c = lstsq(rows, y)
    print("%.10g + %.10g * s + %.10g * log(r)" % (a, b, c))


if __name__ == "__main__":
    main()
