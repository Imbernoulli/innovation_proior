# TIER: greedy
# The obvious recipe: "this is a power law, so fit a power law." Regress
# log(y) directly on log(x1..x4) by ordinary least squares -- five free
# coefficients (intercept + 4 exponents), never touching the grading matrix
# that is sitting right there in the input. This matches the training band
# well (the design is narrow and near-orthogonal). But the true response only
# depends on ONE dimensionless combination of the four knobs; the other three
# directions in log-knob-space have EXACTLY ZERO true effect on y, yet the
# unconstrained fit still spends four degrees of freedom and picks up sample
# noise along all of them. Held out, the knobs are pushed independently over
# a wide range -- mostly a mix of the true direction and large motion
# orthogonal to it -- and the noise picked up in those orthogonal directions
# gets amplified by that large, task-irrelevant motion. The recipe never
# checks whether the grading matrix says some of its four "signals" are pure
# noise.
import sys, math


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
    vals = data[2 + 12:]
    feats = []   # [1, log x1, log x2, log x3, log x4]
    ys = []
    for i in range(n):
        x1 = float(vals[5 * i]); x2 = float(vals[5 * i + 1])
        x3 = float(vals[5 * i + 2]); x4 = float(vals[5 * i + 3])
        y = float(vals[5 * i + 4])
        feats.append([1.0, math.log(x1), math.log(x2), math.log(x3), math.log(x4)])
        ys.append(math.log(y))
    k = 5
    A = [[0.0] * k for _ in range(k)]
    b = [0.0] * k
    for f, yy in zip(feats, ys):
        for r in range(k):
            b[r] += f[r] * yy
            for c in range(k):
                A[r][c] += f[r] * f[c]
    c0, a1, a2, a3, a4 = solve(A, b)
    C = math.exp(c0)
    print("%.10g * x1**%.10g * x2**%.10g * x3**%.10g * x4**%.10g" % (C, a1, a2, a3, a4))


if __name__ == "__main__":
    main()
