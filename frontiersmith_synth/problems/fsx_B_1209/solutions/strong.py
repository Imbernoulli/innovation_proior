# TIER: strong
# The insight: G looks flat only because latent heat buffers it -- the
# forcing f is still carrying real information. Reconstruct the SAME floored
# running-sum accumulator the hidden physics uses (ACC relu(Ak1+f)) instead
# of regressing on G directly. From the training accumulator we can only
# ever see a LOWER bound on the hidden capacity (training never crosses it by
# construction) -- so assume the true capacity is roughly double the largest
# accumulated surplus observed, and emit a program that compares the LIVE,
# rolled-forward accumulator against the *remaining* budget from the end of
# training. Once the live accumulator clears that budget, shape the response
# with a saturating Stefan-sqrt curve (capped at the known active-layer
# bound) rather than an unbounded one. This is what catches the early-
# crossing trap stations that a memoryless fit of G cannot see coming --
# though the insulation feedback and the exact capacity are never visible in
# training, so real residual error remains.
import sys, math

DMAX = 3.0          # active-layer saturation bound (stated in the problem)
KAPPA_GUESS = 0.62   # generic Stefan growth-rate guess
EK_GUESS = 0.65      # generic response-amplitude guess
CAPACITY_MARGIN = 2.0  # assume true capacity ~ 2x the largest observed surplus


def main():
    data = sys.stdin.read().split()
    if not data:
        print("OUT 0.0"); return
    n = int(data[0])
    vals = data[2:]
    f = [float(vals[2 * i]) for i in range(n)]
    g = [float(vals[2 * i + 1]) for i in range(n)]
    if not f:
        print("OUT 0.0"); return

    mean_g = sum(g) / len(g)

    e = 0.0
    max_e = 0.0
    for x in f:
        e = max(0.0, e + x)
        max_e = max(max_e, e)
    e_train_final = e

    lh_hat_abs = CAPACITY_MARGIN * max_e
    lh_hat_eff = max(0.0, lh_hat_abs - e_train_final)

    print("ACC relu ( Ak1 + f )")
    print("OUT %.6f + %.6f * ( %.6f * tanh ( ( %.6f * sqrt ( relu ( A - %.6f ) ) ) / %.6f ) )"
          % (mean_g, EK_GUESS, DMAX, KAPPA_GUESS, lh_hat_eff, DMAX))


if __name__ == "__main__":
    main()
