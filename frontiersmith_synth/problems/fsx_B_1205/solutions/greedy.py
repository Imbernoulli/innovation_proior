# TIER: greedy
# The obvious recipe: an average strong coder sees three noisy training
# columns (G, H, y) and reaches for ordinary least squares over BOTH
# regressors:
#   y = a0 + a1*G + a2*H
# fit via the closed-form normal equations on the historical log. This
# is a perfectly reasonable multi-variable regression and it captures a
# real, if statistically diluted, correlation: whenever the log's
# incidental "ordinary weather" heat blips happen to graze the flowering
# window, yield dips a little, so a2 comes out negative. Nothing here is
# wrong on its face -- it beats a G-only fit on the training log itself.
#
# But the fitted model is LINEAR in H, while the true agronomic law
# penalizes flowering-window heat exceedance QUADRATICALLY (a short
# spike is far worse than the model's slope predicts), and a2 was
# estimated almost entirely from small, incidental H values that were
# rarely large enough to move yield much either way. Extrapolated to the
# held-out heat wave -- deliberately timed to land squarely on that
# season's flowering window, producing H values many times larger than
# anything in training -- this linear model badly under-predicts the
# damage.
import sys


def ols3(rows):
    n = len(rows)
    sG = sH = sY = sGG = sHH = sGH = sGY = sHY = 0.0
    for G, H, y in rows:
        sG += G; sH += H; sY += y
        sGG += G * G; sHH += H * H; sGH += G * H
        sGY += G * y; sHY += H * y
    A = [[float(n), sG, sH], [sG, sGG, sGH], [sH, sGH, sHH]]
    Bv = [sY, sGY, sHY]
    for i in range(3):
        A[i][i] += 1e-6  # tiny ridge for numerical safety

    def det3(m):
        return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))

    D = det3(A)
    if abs(D) < 1e-9:
        return sum(y for _, _, y in rows) / n, 0.0, 0.0

    def repl(col):
        M = [row[:] for row in A]
        for i in range(3):
            M[i][col] = Bv[i]
        return det3(M)

    a0 = repl(0) / D
    a1 = repl(1) / D
    a2 = repl(2) / D
    return a0, a1, a2


def main():
    data = sys.stdin.read().split()
    if len(data) < 2:
        print("40.0")
        return
    n_train = int(data[1])
    vals = data[3:]  # skip testId, n_train, BETA
    rows = []
    for i in range(n_train):
        G = float(vals[3 * i])
        H = float(vals[3 * i + 1])
        y = float(vals[3 * i + 2])
        rows.append((G, H, y))

    a0, a1, a2 = ols3(rows)

    terms = ["%.6f" % a0]
    terms += ["+", "%.6f" % a1, "*", "G"]
    if a2 >= 0:
        terms += ["+", "%.6f" % a2, "*", "H"]
    else:
        terms += ["-", "%.6f" % abs(a2), "*", "H"]
    print(" ".join(terms))


if __name__ == "__main__":
    main()
