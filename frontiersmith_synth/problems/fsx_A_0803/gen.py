#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy magnetization notebook to stdout.

A ferromagnet is measured near its hidden Curie temperature Tc.  The order
parameter (magnetization magnitude) m depends on the temperature T and the
applied field h through a SINGLE hidden scaling law of the form

    m(T, h) = A * |Tc - T|^beta * F( h / |Tc - T|^phi ),   F(x) = (1+x^2)^(-p)

with p = beta / (2*phi) fixed by consistency at T = Tc (the critical isotherm
m ~ h^(1/delta) must be T-independent there).  |Tc-T|^beta is NON-ANALYTIC at
T = Tc whenever beta is not an integer -- there is no smooth (polynomial /
exponential) function of T that agrees with it on both sides of the
transition.

The TRAIN notebook is what the solver SEES: a single campaign that measures
the sample on only ONE side of the transition (T < Tc), sweeping the field h
widely but keeping the distance to Tc in a MODERATE band -- the campaign never
gets close to the transition and never crosses it.  The HELD-OUT grading grid
(regenerated ONLY inside the grader) places most points MUCH closer to Tc than
any training point, on BOTH sides of the transition (including the previously
unseen T > Tc side).  A smooth extrapolation in T that fits the training band
well has no way to reproduce the cusp at Tc or the fold onto the other side.

STDOUT prints ONLY: header "<n> <test_id>" then n rows "<T> <h> <m>".
The hidden Tc, exponents, amplitude and seeds are NEVER printed.
"""
import sys, math, random

# ---- fixed design constants (mirrored byte-for-byte in verify.py) ----
USWEEP = math.log(1.6)     # moderate one-sided distance-to-Tc sweep (train stays away from Tc)
HSWEEP = math.log(15.0)    # field h swept widely (primary axis)
NOISE_TRAIN = 0.04         # multiplicative lognormal sensor noise (train)
N_TRAIN = 180


def params(t):
    """Hidden critical law for this test id (identical in gen.py and verify.py)."""
    rng = random.Random(511000 + t * 8123457)
    Tc = rng.uniform(1.60, 2.40)
    beta = rng.uniform(0.25, 0.55)
    phi = rng.uniform(1.00, 2.20)
    A = rng.uniform(0.80, 1.60)
    U0 = math.exp(rng.uniform(math.log(0.35), math.log(0.55)))
    H0 = math.exp(rng.uniform(math.log(0.05), math.log(0.40)))
    return Tc, beta, phi, A, U0, H0


def true_m(T, h, Tc, beta, phi, A):
    u = abs(Tc - T)
    if u < 1e-9:
        u = 1e-9
    x = h / (u ** phi)
    p = beta / (2.0 * phi)
    F = (1.0 + x * x) ** (-p)
    return A * (u ** beta) * F


def gen_train(t):
    Tc, beta, phi, A, U0, H0 = params(t)
    rng = random.Random(191 + t * 17)
    rows = []
    for _ in range(N_TRAIN):
        u = U0 * math.exp(rng.uniform(-USWEEP, USWEEP))
        T = Tc - u                      # ONE side only: every training point has T < Tc
        h = H0 * math.exp(rng.uniform(-HSWEEP, HSWEEP))
        m = true_m(T, h, Tc, beta, phi, A) * math.exp(rng.gauss(0.0, NOISE_TRAIN))
        rows.append((T, h, m))
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = gen_train(t)
    out = ["%d %d" % (len(rows), t)]
    for T, h, m in rows:
        out.append("%.8g %.8g %.8g" % (T, h, m))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
