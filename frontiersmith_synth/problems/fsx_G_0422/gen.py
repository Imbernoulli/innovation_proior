#!/usr/bin/env python3
# gen.py -- prints ONE instance (the TRAIN stress-strain sample) for a given testId to stdout.
#   python3 gen.py <testId>       testId = 1..N  (difficulty ladder: measurement noise rises)
#
# Physical setting: uniaxial tensile "stress test" of a metal alloy coupon.  We measure the
# true (Cauchy) flow stress sigma [MPa] at a grid of plastic strains s and strain rates r.
# The hidden constitutive law + measurement-noise seed live HERE and (byte-identically) inside
# verify.py.  This file prints DATA ROWS ONLY -- no seed, no law, no coefficients -- so the
# solver must DISCOVER the functional form from the numbers alone.
#
# Hidden constitutive law (NOT printed):
#     sigma(s, r) = Y + K * s**n + C * r**q        (+ multiplicative measurement noise)
#   * Y          : yield / floor stress
#   * K * s**n   : Ludwik strain-hardening term (n<1 -> concave, hardening rate decays)
#   * C * r**q   : strain-rate hardening term
# Coefficients (Y,K,n,C,q) and noise sigma depend deterministically on testId.
import sys, random

# ---- FIXED sampling grids (identical in verify.py) ----
# Train: LOW plastic strain, LOW strain rate (the coupon's mild, safe operating envelope).
S_TRAIN = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
R_TRAIN = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
# Held-out: HIGH strain AND HIGH rate (the coupon pushed toward necking / impact) -- grader only.
S_HOLD  = [0.15, 0.22, 0.30, 0.40]
R_HOLD  = [64.0, 128.0, 256.0, 512.0]


def law_params(test_id):
    """Deterministic ground-truth coefficients for this instance (identical in verify.py)."""
    rr = random.Random(4220 + test_id)
    Y = rr.uniform(180.0, 320.0)    # yield / floor stress [MPa]
    K = rr.uniform(400.0, 900.0)    # strain-hardening coefficient
    n = rr.uniform(0.30, 0.60)      # Ludwik hardening exponent (concave, <1)
    C = rr.uniform(20.0, 60.0)      # strain-rate hardening coefficient
    q = rr.uniform(0.08, 0.20)      # rate-hardening exponent
    return Y, K, n, C, q


def true_val(p, s, r):
    Y, K, n, C, q = p
    return Y + K * (s ** n) + C * (r ** q)


def sigma_for(test_id):
    return 0.020 + 0.004 * test_id


def gen_points(test_id, Ss, Rs, tag):
    """Regenerate the noisy (s,r,sigma) points. tag=1 -> train, tag=2 -> held-out."""
    p = law_params(test_id)
    sig = sigma_for(test_id)
    rng = random.Random(77000 + test_id * 13 + tag)
    pts = []
    for s in Ss:
        for r in Rs:
            val = true_val(p, s, r) * (1.0 + rng.gauss(0.0, sig))
            pts.append((s, r, val))
    return pts


def main():
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    train = gen_points(test_id, S_TRAIN, R_TRAIN, tag=1)
    out = [str(len(train))]
    for (s, r, val) in train:
        out.append("%.12g %.12g %.12g" % (s, r, val))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
