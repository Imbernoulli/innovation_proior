# TIER: strong
# The insight: RSS assumes parts are independent, but same-batch parts share
# a common-mode bias, so the total variance is a COMPOSITION --
# independent-part variance PLUS a correlated same-batch term --
#     D^2 = S2 + TAU2 * B2,      TAU2 >= 0 hidden.
# We do not fit a free power law in n/S2/B2 (that hypothesis space cannot
# represent an additive-under-sqrt composition). Instead we isolate the
# residual R_i = D_i^2 - S2_i, which the composition law says should equal
# TAU2 * B2_i plus noise, and recover TAU2 by a 1-D least-squares fit of R
# against B2, forced through the origin (physically TAU2*0 = 0) and clipped
# non-negative (a variance contribution cannot be negative). The per-row
# signal is tiny -- swamped by pilot-line sensor noise -- but averaging the
# fit over many rows recovers TAU2 well enough to extrapolate correctly to
# the large-batch held-out regime where the RSS recipe diverges.
import sys, math


def main():
    data = sys.stdin.read().split()
    if not data:
        print("1.0"); return
    rows = int(data[0])
    vals = data[2:]
    num = 0.0
    den = 0.0
    for i in range(rows):
        S2 = float(vals[5 * i + 2])
        B2 = float(vals[5 * i + 3])
        D = float(vals[5 * i + 4])
        R = D * D - S2
        num += B2 * R
        den += B2 * B2
    tau2 = num / den if den > 1e-18 else 0.0
    if tau2 < 0.0:
        tau2 = 0.0
    print("sqrt(S2 + %.10g * B2)" % tau2)


if __name__ == "__main__":
    main()
