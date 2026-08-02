# TIER: greedy
# The obvious recipe: an average strong coder sees three noisy training
# columns (T, S, y) and reaches for ordinary least squares. Fit
#   y = a + b*T + c*S
# on the calm log via the closed-form normal equations. This captures the
# LINEAR superposition of tide and surge almost perfectly on calm data --
# the fitted (a,b,c) come out very close to (0,1,1), because on calm data
# the kappa*T*S interaction term is far smaller than the sensor noise and
# therefore has essentially no leverage on the fit. The header hands us
# an interaction coefficient kappa, but nothing in the training residuals
# ever correlates with it, so a fit-only approach has no statistical
# reason to use it -- and the resulting additive model over-predicts
# whenever a large surge coincides with a high tide (exactly the
# held-out storm).
import sys


def ols3(rows):
    n = len(rows)
    sT = sS = sY = sTT = sSS = sTS = sTY = sSY = 0.0
    for T, S, y in rows:
        sT += T; sS += S; sY += y
        sTT += T * T; sSS += S * S; sTS += T * S
        sTY += T * y; sSY += S * y
    A = [[float(n), sT, sS], [sT, sTT, sTS], [sS, sTS, sSS]]
    Bv = [sY, sTY, sSY]
    for i in range(3):
        A[i][i] += 1e-9  # tiny ridge for numerical safety

    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    D = det3(A)
    if abs(D) < 1e-12:
        return 0.0, 1.0, 1.0

    def repl(col):
        M = [row[:] for row in A]
        for i in range(3):
            M[i][col] = Bv[i]
        return det3(M)

    a = repl(0) / D
    b = repl(1) / D
    c = repl(2) / D
    return a, b, c


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("T")
        return
    n_train = int(data[1])
    vals = data[3:]  # skip testId, n_train, kappa
    rows = []
    for i in range(n_train):
        T = float(vals[3 * i])
        S = float(vals[3 * i + 1])
        y = float(vals[3 * i + 2])
        rows.append((T, S, y))

    a, b, c = ols3(rows)

    terms = ["%.6f" % a]
    if b >= 0:
        terms += ["+", "%.6f" % b, "*", "T"]
    else:
        terms += ["-", "%.6f" % abs(b), "*", "T"]
    if c >= 0:
        terms += ["+", "%.6f" % c, "*", "S"]
    else:
        terms += ["-", "%.6f" % abs(c), "*", "S"]
    print(" ".join(terms))


if __name__ == "__main__":
    main()
