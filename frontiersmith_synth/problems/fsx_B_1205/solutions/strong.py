# TIER: strong
# The insight: the historical log cannot statistically PIN DOWN the
# flowering-stress coefficient (H is almost always tiny in ordinary
# seasons, so BETA*H**2 is swamped by measurement noise there -- see
# greedy.py), but it does not need to be learned from residuals. It is
# handed to us directly, as an independently surveyed physiological
# constant. The stated mechanism says the penalty is a SQUARE of the
# flowering-window heat exceedance, stage-weighted rather than a plain
# season aggregate -- not a linear nuisance term folded into a generic
# regression.
#
# The exchange this problem is testing: first UNDO the known quadratic
# penalty on the training targets (y + BETA*H**2 -- what yield would
# have been with zero flowering stress), fit the simple, honest
# season-total-GDD relationship on those corrected targets (this is
# where the real, well-conditioned training signal lives), and then
# RE-APPLY the given penalty when predicting -- trusting the survey
# constant instead of discarding it for lack of training-time leverage.
import sys


def ols_g(rows):
    n = len(rows)
    sG = sY = sGG = sGY = 0.0
    for G, y in rows:
        sG += G; sY += y; sGG += G * G; sGY += G * y
    denom = n * sGG - sG * sG
    if abs(denom) < 1e-9:
        return sum(y for _, y in rows) / n, 0.0
    a1 = (n * sGY - sG * sY) / denom
    a0 = (sY - a1 * sG) / n
    return a0, a1


def main():
    data = sys.stdin.read().split()
    if len(data) < 3:
        print("40.0")
        return
    n_train = int(data[1])
    BETA = float(data[2])
    vals = data[3:]
    rows = []
    for i in range(n_train):
        G = float(vals[3 * i])
        H = float(vals[3 * i + 1])
        y = float(vals[3 * i + 2])
        rows.append((G, y + BETA * H * H))  # undo the known penalty first

    a0, a1 = ols_g(rows)

    print("%.6f + %.6f * G - %.6f * H ** 2" % (a0, a1, BETA))


if __name__ == "__main__":
    main()
