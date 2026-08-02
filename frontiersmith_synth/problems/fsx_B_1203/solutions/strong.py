# TIER: strong
# The insight: kappa cannot be LEARNED from the calm log (its effect on y
# is statistically invisible there -- see greedy.py), but it does not
# need to be learned. It is handed to us directly as a bathymetry survey
# constant. The stated mechanism says the tide and surge interact
# nonlinearly and are NOT simply additive; the honest model is the full
# interaction expansion T + S - kappa*T*S. Reading and trusting the given
# kappa -- rather than discarding it as a column with no training-time
# statistical leverage, as a pure-fit approach does -- is exactly the
# exchange this problem is testing.
import sys


def main():
    data = sys.stdin.read().split()
    if len(data) < 3:
        print("T")
        return
    kappa = float(data[2])
    print("T + S - %.6f * T * S" % kappa)


if __name__ == "__main__":
    main()
