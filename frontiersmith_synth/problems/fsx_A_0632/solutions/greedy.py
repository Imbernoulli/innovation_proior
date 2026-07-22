# TIER: greedy
# The obvious recipe: take the two mechanisms the statement spells out at face
# value -- a constant standby loss and a temperature-modulated resistive loss
# proportional to L^2 -- and fit exactly that basis
#     y = c0 + c1*L^2 + c2*T*L^2
# by ordinary least squares. This nails the 20-60% commissioning band (the
# saturation term is only a few percent of the signal there, well inside the
# noise a plain quadratic fit shrugs off), so a practitioner who checks the
# training residual and sees it small stops here. But no term in this basis
# can grow faster than L^2, so it structurally cannot express the core's
# faster-than-quadratic saturation. On the 80-110% overload grid, where that
# term can exceed the resistive loss itself, this law drifts increasingly
# wrong as load rises.
import sys


def solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
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
        print("0.0"); return
    n = int(data[0])
    vals = data[2:]
    feats = []  # [1, L^2, T*L^2]
    y = []
    for i in range(n):
        L = float(vals[3 * i]); T = float(vals[3 * i + 1]); yy = float(vals[3 * i + 2])
        feats.append([1.0, L * L, T * L * L])
        y.append(yy)
    m = 3
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for x, yy in zip(feats, y):
        for r in range(m):
            b[r] += x[r] * yy
            for c in range(m):
                A[r][c] += x[r] * x[c]
    c0, c1, c2 = solve(A, b)
    print("%.10g + %.10g * L**2 + %.10g * T*L**2" % (c0, c1, c2))


if __name__ == "__main__":
    main()
