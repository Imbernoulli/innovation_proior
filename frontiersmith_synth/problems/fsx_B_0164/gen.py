#!/usr/bin/env python3
# gen.py -- prints ONE instance (the TRAIN table) for a given testId to stdout.
#   python3 gen.py <testId>       testId = 1..N  (difficulty ladder: noise rises with testId)
#
# The hidden scaling law and the measurement-noise seed live here AND (independently,
# byte-identical) inside verify.py.  This file prints DATA ROWS ONLY -- no seed, no law,
# no coefficients -- so the solver must discover the functional form from the numbers.
#
# Hidden law (NOT printed):  D(T,V) = E + A*T**(-alpha) + B*V**beta   (+ multiplicative noise)
# Coefficients (E,A,alpha,B,beta) and noise sigma depend deterministically on testId.
import sys, random, math

# ---- FIXED sampling grids (identical in verify.py) ----
T_TRAIN = [4, 5, 6, 7, 8, 9, 10, 12]
V_TRAIN = [2, 3, 4, 5, 6, 8]
# held-out (busier future yard): strictly larger tracks AND larger volume -- used only by grader
T_HOLD  = [13, 15, 18, 22]
V_HOLD  = [9, 11, 14, 18]


def law_params(test_id):
    """Deterministic ground-truth coefficients for this instance (identical in verify.py)."""
    r = random.Random(1000 + test_id)
    E     = r.uniform(0.5, 1.2)     # irreducible dwell floor
    A     = r.uniform(4.0, 9.0)     # track-parallelism gain
    alpha = r.uniform(0.6, 1.0)     # decay exponent in T
    B     = r.uniform(0.08, 0.25)   # congestion coefficient
    beta  = r.uniform(1.35, 1.9)    # super-linear growth exponent in V
    return E, A, alpha, B, beta


def true_val(p, T, V):
    E, A, alpha, B, beta = p
    return E + A * (T ** (-alpha)) + B * (V ** beta)


def sigma_for(test_id):
    return 0.055 + 0.006 * test_id


def gen_points(test_id, Ts, Vs, tag):
    """Regenerate the noisy (T,V,D) points. tag=1 -> train, tag=2 -> held-out."""
    p = law_params(test_id)
    sig = sigma_for(test_id)
    rng = random.Random(9000 + test_id * 7 + tag)
    pts = []
    for t in Ts:
        for v in Vs:
            d = true_val(p, t, v) * (1.0 + rng.gauss(0.0, sig))
            pts.append((t, v, d))
    return pts


def main():
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    train = gen_points(test_id, T_TRAIN, V_TRAIN, tag=1)
    out = [str(len(train))]
    for (t, v, d) in train:
        # full precision so the grader can re-identify the instance exactly
        out.append("%d %d %.12g" % (t, v, d))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
