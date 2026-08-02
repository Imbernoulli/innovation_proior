# TIER: greedy
# The obvious move: read the calibration burst, and just SOLVE for the
# inverse filter over the FULL 15-tap window (radius 7) by ordinary least
# squares -- no regularisation, no attempt to ask whether the calibration
# burst actually supports 15 free parameters. On wide-kernel / long-burst
# tests this looks like a perfectly reasonable "textbook" deconvolution
# filter. On narrow-kernel / short-burst tests the regression sits right at
# (or past) the interpolation threshold and the fitted taps blow up,
# amplifying fresh noise on the held-out trace far worse than doing nothing.
import sys
import numpy as np

R = 7
TAPS = list(range(-R, R + 1))


def tap_name(j):
    if j < 0:
        return "ym%d" % (-j)
    if j == 0:
        return "y0"
    return "yp%d" % j


def main():
    data = sys.stdin.read().split()
    if len(data) < 3:
        print("0"); return
    ncal = int(data[0])
    vals = data[3:]
    x_cal = [0.0] * ncal
    y_cal = [0.0] * ncal
    for i in range(ncal):
        x_cal[i] = float(vals[2 * i])
        y_cal[i] = float(vals[2 * i + 1])

    lo, hi = R, ncal - R
    if hi <= lo:
        print("0"); return

    rows = []
    targets = []
    for i in range(lo, hi):
        rows.append([y_cal[i + j] for j in TAPS])
        targets.append(x_cal[i])
    A = np.array(rows, dtype=float)
    b = np.array(targets, dtype=float)

    coef, *_ = np.linalg.lstsq(A, b, rcond=None)

    terms = []
    for j, c in zip(TAPS, coef):
        terms.append("%.6f * %s" % (float(c), tap_name(j)))
    print(" + ".join(terms))


if __name__ == "__main__":
    main()
