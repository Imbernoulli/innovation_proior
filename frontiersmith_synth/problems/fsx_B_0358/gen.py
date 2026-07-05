#!/usr/bin/env python3
# gen.py -- prints ONE instance (the TRAIN table) for a given testId to stdout.
#   python3 gen.py <testId>       testId = 1..N  (difficulty ladder: noise rises with testId)
#
# Theme: ROOFTOP GARDENS.  Each instance is a monitoring table from a set of small pilot
# rooftop beds.  For a bed of soil area A (m^2) whose canopy runs H degrees C hotter than the
# shaded ambient at midday, the irrigation controller logs the total daily water DEMAND D
# (litres/day) needed to keep the canopy un-stressed.  Bigger, hotter beds need more water,
# and the demand grows FASTER than linearly as area and heat pile up together.
#
# The hidden scaling law and the measurement-noise seed live here AND (independently,
# byte-identical) inside verify.py.  This file prints DATA ROWS ONLY -- no seed, no law,
# no coefficients -- so the solver must discover the functional form from the numbers.
#
# Hidden law (NOT printed):  D(A,H) = D0 + C * (A + rho*H) ** q     (+ multiplicative noise)
#   -- a COMBINED-RESOURCE power law with a SUPER-LINEAR exponent q>1: area and heat pool into
#      one effective demand-driver (A + rho*H) that is raised to a growth exponent, atop a fixed
#      baseline draw D0.  The pooled, super-linear structure is what makes the busier-rooftop
#      extrapolation hard for a separable product power law.
# Coefficients (D0, C, rho, q) and noise sigma depend deterministically on testId.
import sys, random, math

# ---- FIXED sampling grids (identical in verify.py) ----
A_TRAIN = [6, 8, 10, 13, 17, 22, 28, 36]     # bed soil area (m^2), small pilot beds
H_TRAIN = [4, 6, 9, 13, 18, 25]              # midday canopy heat excess (deg C over shade)
# held-out (larger + hotter future rooftop farm): strictly larger A AND H -- grader only
A_HOLD  = [55, 75, 100, 140]
H_HOLD  = [38, 55, 80, 120]


def law_params(test_id):
    """Deterministic ground-truth coefficients for this instance (identical in verify.py)."""
    r = random.Random(4200 + test_id)
    D0  = r.uniform(5.0, 20.0)      # fixed baseline draw (pumps, evaporation offset)
    C   = r.uniform(0.30, 1.00)     # demand amplitude
    rho = r.uniform(0.8, 2.0)       # deg-C-to-area equivalence in the pooled driver
    q   = r.uniform(1.20, 1.60)     # super-linear growth exponent (>1)
    return D0, C, rho, q


def true_val(pp, A, H):
    D0, C, rho, q = pp
    return D0 + C * ((A + rho * H) ** q)


def sigma_for(test_id):
    return 0.030 + 0.004 * test_id


def gen_points(test_id, As, Hs, tag):
    """Regenerate the noisy (A,H,D) points. tag=1 -> train, tag=2 -> held-out."""
    pp = law_params(test_id)
    sig = sigma_for(test_id)
    rng = random.Random(77000 + test_id * 13 + tag)
    pts = []
    for a in As:
        for h in Hs:
            d = true_val(pp, a, h) * (1.0 + rng.gauss(0.0, sig))
            pts.append((a, h, d))
    return pts


def main():
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    train = gen_points(test_id, A_TRAIN, H_TRAIN, tag=1)
    out = [str(len(train))]
    for (a, h, d) in train:
        # full precision so the grader can re-identify the instance exactly
        out.append("%d %d %.12g" % (a, h, d))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
