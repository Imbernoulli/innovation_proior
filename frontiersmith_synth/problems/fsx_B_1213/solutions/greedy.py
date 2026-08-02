# TIER: greedy
# The obvious recipe: no domain knowledge about receptor saturation or
# accelerating toxicity -- just fit a generic quadratic (least squares) to
# each observed curve and extrapolate it across the full allowed range. This
# is what a strong coder without pharmacology intuition reaches for first: a
# numpy.polyfit curve fit. A quadratic has no ceiling, so on training windows
# where efficacy is still rising it usually keeps "seeing" efficacy climb
# across the whole dosing range (its parabola vertex, if any, often sits past
# Dmax), while its fit of toxicity's acceleration is only as good as a
# quadratic can extrapolate a curve of unknown, faster-than-quadratic growth.
# On several test cases this systematically pushes the recommended dose to
# (or near) the ethical dosing ceiling -- exactly where the true therapeutic
# window has already closed.
import sys
import numpy as np


def main():
    data = sys.stdin.read().split()
    if len(data) < 3:
        print("EFFICACY 0")
        print("TOXICITY 0")
        return
    n = int(data[0])
    vals = data[3:]
    d = np.array([float(vals[3 * i]) for i in range(n)])
    e = np.array([float(vals[3 * i + 1]) for i in range(n)])
    tx = np.array([float(vals[3 * i + 2]) for i in range(n)])

    ce = np.polyfit(d, e, 2)
    ct = np.polyfit(d, tx, 2)

    def fmt(coefs):
        # coefs = [a2, a1, a0] for a2*d^2 + a1*d + a0
        # NOTE: numeric coefficients are kept as their own whitespace-separated
        # tokens (not glued to an operator) so the expression is not
        # accidentally robust to token-level corruption of its constants.
        a2, a1, a0 = coefs
        return "%.8g * d**2 + %.8g * d + %.8g" % (a2, a1, a0)

    print("EFFICACY %s" % fmt(ce))
    print("TOXICITY %s" % fmt(ct))


if __name__ == "__main__":
    main()
