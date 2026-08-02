#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy bridge-deck monitoring log to stdout.

Hidden displacement law (identical copy lives in verify.py; this file NEVER
prints it, its coefficients, or the seed):

    T_true(t)   = Tmean + Tamp * cos(2*pi*(t-phase)/P)              seasonal deck temperature
    d_creep(t)  = Dinf * (1 - exp(-t/tau)) + drift * t               IRREVERSIBLE settlement
    d_thermal(t)= alpha * (T_true(t) - Tmean)                        REVERSIBLE thermal swing
    d_obs(t)    = d_creep(t) + d_thermal(t) + noise                  what the sensor logs

The solver sees a monitoring log confined to the VISIBLE span t in [0, T_END_VIS)
-- a temperature sensor reading T (noisy) and a displacement reading d (noisy)
at each of N_TRAIN roughly-evenly-spaced ticks. The held-out grading horizon
(several seasonal cycles later) is generated ONLY inside verify.py.

For 7 of the 10 test ids the visible window is engineered to END almost exactly
at a seasonal temperature PEAK (phase chosen so T_true(T_END_VIS) ~= Tmean+Tamp):
the last few logged readings are elevated purely by reversible thermal expansion,
not by ongoing settlement -- the trap for any method that fits the raw
displacement trace as if it were pure trend.

STDOUT prints ONLY: header "<n_train> <id>" then n_train rows "<t> <T> <d>".
"""
import sys, math, random

# ---- fixed design constants (byte-for-byte mirrored in verify.py) ----
P = 365.0                 # seasonal period, days (annual cycle)
T_END_VIS = 1200.0        # visible monitoring span: t in [0, T_END_VIS)
N_TRAIN = 200
TRAP_IDS = {1, 2, 3, 4, 5, 6, 7}   # cases whose visible window ends at a seasonal peak


def params(t):
    """Hidden displacement law for this test id (identical in gen.py / verify.py)."""
    rng = random.Random(700000 + t * 104729)
    Dinf = rng.uniform(35.0, 65.0)      # mm, magnitude of the settling (saturating) term
    tau = rng.uniform(500.0, 950.0)     # days, settlement relaxation time
    drift = rng.uniform(0.01, 0.025)    # mm/day, slow secondary creep that never saturates
    alpha = rng.uniform(0.9, 1.8)       # mm per deg C, reversible thermal sensitivity
    Tmean = rng.uniform(8.0, 18.0)      # deg C, mean deck temperature
    Tamp = rng.uniform(8.0, 15.0)       # deg C, seasonal amplitude
    if t in TRAP_IDS:
        phase = (T_END_VIS - rng.uniform(-5.0, 5.0)) % P   # window ends near a seasonal peak
    else:
        phase = rng.uniform(0.0, P)                        # unconstrained phase
    sigma_T = rng.uniform(1.2, 2.2)     # deg C, train temperature sensor noise (stdev)
    sigma_d = rng.uniform(4.5, 8.0)     # mm, train displacement sensor noise (stdev)
    return Dinf, tau, drift, alpha, Tmean, Tamp, phase, sigma_T, sigma_d


def T_true(t, Tmean, Tamp, phase):
    return Tmean + Tamp * math.cos(2.0 * math.pi * (t - phase) / P)


def d_creep(t, Dinf, tau, drift):
    return Dinf * (1.0 - math.exp(-t / tau)) + drift * t


def d_thermal(t, alpha, Tmean, Tamp, phase):
    return alpha * (T_true(t, Tmean, Tamp, phase) - Tmean)


def gen_train(t):
    Dinf, tau, drift, alpha, Tmean, Tamp, phase, sigma_T, sigma_d = params(t)
    rng = random.Random(1000 + t * 13)
    rows = []
    for i in range(N_TRAIN):
        tt = i * (T_END_VIS / N_TRAIN)
        Traw = T_true(tt, Tmean, Tamp, phase) + rng.gauss(0.0, sigma_T)
        d = (d_creep(tt, Dinf, tau, drift)
             + d_thermal(tt, alpha, Tmean, Tamp, phase)
             + rng.gauss(0.0, sigma_d))
        rows.append((tt, Traw, d))
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = gen_train(t)
    out = ["%d %d" % (len(rows), t)]
    for tt, Traw, d in rows:
        out.append("%.8g %.8g %.8g" % (tt, Traw, d))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
