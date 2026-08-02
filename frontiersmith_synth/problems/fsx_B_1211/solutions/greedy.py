# TIER: greedy
# The obvious physical recipe: assume the rift keeps growing at its OBSERVED
# phase-1 rate forever (never notice the buttressing-loss speed-up) and invert
# the single-phase relation
#     T = (kappa*H0 - D0) / (c0 + kappa*gamma)
#   =>  kappa = (D0 + T*c0) / (H0 - T*gamma)
# per training row to estimate the shelf's critical ratio kappa, then average
# across rows and emit the single-phase closed form with that kappa baked in.
#
# Training segments were logged with phi just below kappa, so buttressing loss
# barely engages before they calve -- this fit looks accurate in-sample. But it
# never uses phi at all, so on held-out segments (which spend almost their
# whole life in the accelerated phase-2 regime) it keeps assuming the slow
# phase-1 rate and predicts calving far too late.
import sys


def main():
    data = sys.stdin.read().split()
    if not data:
        print("0.0"); return
    n = int(data[0])
    vals = data[2:]
    kappas = []
    for i in range(n):
        H0 = float(vals[6 * i]); D0 = float(vals[6 * i + 1])
        c0 = float(vals[6 * i + 2]); gamma = float(vals[6 * i + 3])
        T = float(vals[6 * i + 5])
        denom = H0 - T * gamma
        if abs(denom) < 1e-9:
            continue
        kappas.append((D0 + T * c0) / denom)
    if not kappas:
        print("300.0"); return
    kappa_hat = sum(kappas) / len(kappas)
    print("( %.10g * H0 - D0 ) / ( c0 + %.10g * gamma )" % (kappa_hat, kappa_hat))


if __name__ == "__main__":
    main()
