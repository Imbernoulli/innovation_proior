#!/usr/bin/env python3
"""
gen.py <testId>   ->  prints ONE turbulence-lab instance (TRAIN sample) to stdout.

Difficulty ladder is keyed ONLY by <testId> (1..N): higher id => more measurement
noise, fewer training rows, and a wider held-out extrapolation gap.

The instance is a table of anonymized turbulence-probe channels x0..x5 measured
across many experiments spanning a range of Reynolds numbers.  Exactly which
power-law combination of channels is (nearly) invariant is NOT revealed here --
the ground-truth generative law lives ONLY inside the checker.  This stdout emits
DATA ROWS ONLY (plus the case id + shape); it never prints the seed, the law, or
any coefficient.

The channels (hidden meaning, for the author only):
    x0 = rms velocity U        [m/s]
    x1 = integral scale L      [m]
    x2 = dissipation eps       [m^2/s^3]
    x3 = kinematic visc nu     [m^2/s]
    x4 = grid/mesh proxy M     [m]     (distractor)
    x5 = ancillary channel W             (distractor)
"""
import sys, math, random

NCOLS = 6

# ---- physical / generative constants (shared byte-for-byte with verify.py) ----
CINF = 0.5          # asymptotic dissipation constant C_eps (zeroth law)
ACORR = 5.0         # finite-Reynolds correction amplitude
SIGMA_G = 0.20      # shared latent (couples eps<->M on TRAIN only)
U_LO, U_HI = 0.05, 5.0
L_LO, L_HI = 0.01, 1.0
M_LO, M_HI = 0.001, 0.05
W_LO, W_HI = 1.0, 100.0
RE_TRAIN_LO, RE_TRAIN_HI = 1.0e3, 3.0e4
RE_HELD_LO = 3.0e4
N_HELD = 300


def params(t):
    ntrain = max(90, 250 - 15 * t)
    sigma_meas = 0.03 + 0.005 * t
    re_held_hi = 3.0e4 * (2 + t)
    return ntrain, sigma_meas, re_held_hi


def _uni_log(rng, lo, hi):
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def make_row(rng, re_lo, re_hi, sigma_meas, is_train):
    """One experiment -> 6 noisy channel readings (all strictly positive)."""
    Re = math.exp(rng.uniform(math.log(re_lo), math.log(re_hi)))
    U = _uni_log(rng, U_LO, U_HI)
    L = _uni_log(rng, L_LO, L_HI)
    nu = U * L / Re
    Re_lambda = math.sqrt(15.0 * Re)
    Ceps = CINF * (1.0 + ACORR / Re_lambda)          # weak finite-Re drift
    eps_clean = Ceps * U ** 3 / L
    baseM = _uni_log(rng, M_LO, M_HI)
    baseW = _uni_log(rng, W_LO, W_HI)

    g = rng.gauss(0.0, 1.0)
    if is_train:
        g_eps = g
        g_M = g                                      # eps and M share latent
    else:
        g_eps = rng.gauss(0.0, 1.0)
        g_M = rng.gauss(0.0, 1.0)                     # decoupled off-regime

    eps = eps_clean * math.exp(SIGMA_G * g_eps)
    M = baseM * math.exp(SIGMA_G * g_M)
    W = baseW

    cols = [U, L, eps, nu, M, W]
    # independent multiplicative measurement noise on every channel
    cols = [c * math.exp(sigma_meas * rng.gauss(0.0, 1.0)) for c in cols]
    return cols


def gen_train(t):
    ntrain, sigma_meas, _ = params(t)
    rng = random.Random(100003 * t + 7)
    return [make_row(rng, RE_TRAIN_LO, RE_TRAIN_HI, sigma_meas, True)
            for _ in range(ntrain)]


def gen_heldout(t):
    ntrain, sigma_meas, re_held_hi = params(t)
    rng = random.Random(500009 * t + 13)
    return [make_row(rng, RE_HELD_LO, re_held_hi, sigma_meas, False)
            for _ in range(N_HELD)]


def main():
    t = int(sys.argv[1])
    rows = gen_train(t)
    out = []
    out.append("TESTID %d" % t)
    out.append("%d %d" % (NCOLS, len(rows)))
    out.append(" ".join("x%d" % i for i in range(NCOLS)))
    for r in rows:
        out.append(" ".join(repr(v) for v in r))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
