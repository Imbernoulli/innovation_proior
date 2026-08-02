#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

Family: corrosion-rate-extrapolate.  A steel coupon's corrosion rate R (mm/yr)
is governed by two regimes:
  - PASSIVE: a thin oxide film protects the metal; R is a smooth, low function
    of chloride concentration Cl, temperature T, pH and exposure time tex.
  - ACTIVE (past a chloride threshold Cl_crit(T,pH) that is itself unknown and
    depends on T and pH): the film breaks down locally (pitting) and R jumps by
    orders of magnitude above the passive-regime trend.

Each row also reports D, an electrochemical "repassivation margin" measured on
that same coupon: D = (Cl_crit - Cl) / Cl_crit for the (unknown, per-instance)
threshold.  D > 0 means the film is intact with that fractional margin left.
Every TRAIN coupon here was pulled from service BEFORE its film broke down, so
D > 0 throughout, and (because excess-over-threshold is exactly 0 whenever
D >= 0) R never actually correlates with D in this file -- the correlation
only appears once D goes negative, which happens only in the held-out,
untested chemistry the checker grades against.

Each testId fixes a DIFFERENT hidden material/chemistry (thresholds, kinetics,
noise).  STDOUT prints ONLY a header "<n_train> <test_id>" then n_train rows
"Cl T pH tex D R".  The hidden law, its coefficients, and the RNG seed are
NEVER printed -- they live only inside gen.py and verify.py (duplicated, not
imported, so a submitted solution cannot reach them).
"""
import sys, random, math


def hidden_params(t):
    """Hidden per-instance chemistry/kinetics. Lives in gen.py AND verify.py."""
    rng = random.Random(311007 + t * 7919)
    return dict(
        A0=rng.uniform(0.0008, 0.0022),        # passive-rate prefactor (mm/yr)
        EaR=rng.uniform(3400.0, 5200.0),       # Arrhenius activation const (K)
        n_t=rng.uniform(-0.28, -0.08),         # exposure-time exponent
        b_cl=rng.uniform(0.012, 0.028),        # mild passive-regime Cl coupling
        c_ph=rng.uniform(0.02, 0.06),          # pH-departure sensitivity
        Cl0=rng.uniform(180.0, 320.0),         # threshold prefactor @25C,pH7
        alphaT=rng.uniform(0.014, 0.024),      # threshold's T-sensitivity
        betapH=rng.uniform(0.05, 0.11),        # threshold's pH-sensitivity
        Kjump=rng.uniform(5.0, 12.0),          # breakdown exponential rate
        gamma=rng.uniform(0.5, 1.5),           # breakdown linear pre-factor
        qcurve=rng.uniform(0.7, 1.4),          # breakdown curvature exponent
        sigmaR=rng.uniform(0.03, 0.06),        # multiplicative log-noise on R
        sigmaD=rng.uniform(0.006, 0.015),      # additive noise on D
    )


def cl_crit(T, pH, p):
    return p["Cl0"] * math.exp(-p["alphaT"] * (T - 25.0)) * (1.0 + p["betapH"] * (pH - 7.0))


def rate_true(Cl, T, pH, tex, p):
    Rp = (p["A0"]
          * math.exp(-p["EaR"] * (1.0 / (T + 273.15) - 1.0 / 298.15))
          * (1.0 + p["c_ph"] * (pH - 7.0) ** 2)
          * ((tex + 1.0) ** p["n_t"])
          * (1.0 + p["b_cl"] * Cl))
    cc = cl_crit(T, pH, p)
    excess = max(0.0, (Cl - cc) / cc)
    ec = excess ** p["qcurve"] if excess > 0.0 else 0.0
    return Rp * (1.0 + p["gamma"] * ec) * math.exp(p["Kjump"] * ec)


def margin_true(Cl, T, pH, p):
    cc = cl_crit(T, pH, p)
    return (cc - Cl) / cc


def gen_train(t, p, n):
    rng = random.Random(900 + t * 104729)
    rows = []
    for _ in range(n):
        T = rng.uniform(10.0, 70.0)
        pH = rng.uniform(5.0, 9.0)
        tex = rng.uniform(5.0, 500.0)
        cc = cl_crit(T, pH, p)
        Cl = rng.uniform(0.05 * cc, 0.85 * cc)   # always comfortably passive
        Rt = rate_true(Cl, T, pH, tex, p)
        Robs = Rt * math.exp(rng.gauss(0.0, p["sigmaR"]))
        Dobs = margin_true(Cl, T, pH, p) + rng.gauss(0.0, p["sigmaD"])
        rows.append((Cl, T, pH, tex, Dobs, Robs))
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    p = hidden_params(t)
    n = 46 - 2 * ((t - 1) % 10)   # 46 .. 28, mild difficulty ladder
    rows = gen_train(t, p, n)
    out = ["%d %d" % (n, t)]
    for (Cl, T, pH, tex, D, R) in rows:
        out.append("%.8f %.6f %.6f %.6f %.8f %.10e" % (Cl, T, pH, tex, D, R))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
