#!/usr/bin/env python3
# gen.py -- prints ONE instance (the EARLY-TIME TRAIN series) for a given testId to stdout.
#   python3 gen.py <testId>       testId = 1..N  (difficulty ladder: measurement noise rises with testId)
#
# Theme: CELL-CULTURE BIOREACTOR.  Each instance is an online-probe log from a fed-batch
# mammalian cell culture.  A viable-cell-density (VCD) sensor samples the reactor every 2 hours
# during the EARLY run (lag + exponential + start of deceleration).  The controller must predict
# the LATE saturation plateau (stationary phase, when nutrients deplete and density levels off)
# from those early samples alone -- the late-time readings are withheld (the grader owns them).
#
# The hidden growth law and the measurement-noise seed live here AND (independently,
# byte-identical) inside verify.py.  This file prints DATA ROWS ONLY -- no seed, no law, no
# coefficients -- so the solver must DISCOVER the functional family (a sigmoidal
# logistic/Gompertz-type saturating law) purely from the early numbers.
#
# Hidden law (NOT printed):  N(t) = K * exp( -b * exp( -c * t ) )      (Gompertz growth)
#   -- a saturating S-curve with carrying capacity K (the stationary-phase plateau), shape b
#      (set by the inoculum fraction N0/K = exp(-b)), and specific growth rate c.  The plateau
#      K is only WEAKLY constrained by early data: a naive exponential fit ignores saturation
#      and blows up, so recovering the saturating family is what makes the late-time
#      extrapolation hard.
# Coefficients (K, b, c) and noise sigma depend deterministically on testId.
import sys, random, math

# ---- FIXED sampling grids (identical in verify.py) ----
T_TRAIN = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]   # early run (hours)
T_HOLD  = [40, 46, 52, 58, 64, 72, 80, 90]                              # late saturation plateau (h)


def law_params(test_id):
    """Deterministic ground-truth coefficients for this instance (identical in verify.py)."""
    r = random.Random(6100 + test_id)
    K = r.uniform(30.0, 90.0)     # carrying capacity / stationary plateau (1e6 viable cells / mL)
    b = r.uniform(3.0, 5.0)       # shape: inoculum fraction N0/K = exp(-b)
    c = r.uniform(0.06, 0.10)     # specific growth rate (1/h); training never reaches the plateau
    return K, b, c


def true_val(pp, t):
    K, b, c = pp
    return K * math.exp(-b * math.exp(-c * t))


def sigma_for(test_id):
    return 0.045 + 0.005 * test_id


def gen_points(test_id, ts, tag):
    """Regenerate the noisy (t, N) samples. tag=1 -> train (early), tag=2 -> held-out (late)."""
    pp = law_params(test_id)
    sig = sigma_for(test_id)
    rng = random.Random(90000 + test_id * 17 + tag)
    pts = []
    for t in ts:
        n = true_val(pp, t) * (1.0 + rng.gauss(0.0, sig))
        pts.append((t, n))
    return pts


def main():
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    train = gen_points(test_id, T_TRAIN, tag=1)
    out = [str(len(train))]
    for (t, n) in train:
        # full precision so the grader can re-identify the instance exactly
        out.append("%d %.12g" % (t, n))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
