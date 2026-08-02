# TIER: trivial
# Do-nothing baseline: reproduce the checker's own internal reference
# predictor -- a plain least-squares fit of yield on total season GDD
# (G) ALONE, computed from exactly the training rows given on stdin.
# Never looks at H (the flowering-window heat exceedance) at all.
import sys


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
        y = float(vals[3 * i + 2])
        rows.append((G, y))

    n = len(rows)
    sG = sY = sGG = sGY = 0.0
    for G, y in rows:
        sG += G; sY += y; sGG += G * G; sGY += G * y
    denom = n * sGG - sG * sG
    if abs(denom) < 1e-9:
        a0 = sY / n
        a1 = 0.0
    else:
        a1 = (n * sGY - sG * sY) / denom
        a0 = (sY - a1 * sG) / n

    print("%.6f + %.6f * G" % (a0, a1))


if __name__ == "__main__":
    main()
