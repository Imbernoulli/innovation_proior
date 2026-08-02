#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE thermal-runaway logbook to stdout.

A self-heating device balances Arrhenius-style heat generation
    G(T) = A * exp(b*T)
against cooling that is proportional to (T - Ta) but hard-capped at Hmax:
    R(T,Ta) = min(h*(T-Ta), Hmax)

Below a critical ambient Ta_crit, G(T)=R(T,Ta) has a stable steady-state
solution.  Above Ta_crit, generation outruns cooling even at full capacity Hmax
and no steady state exists: the device runs away to a fixed protection cutoff
Tfail.  Ta_crit is exactly the ambient at which generation first reaches Hmax
at the "elbow" temperature Ta + Hmax/h:  A*exp(b*(Ta+Hmax/h)) = Hmax.

The TRAIN log the solver sees samples ONLY sub-critical ambients (biased
toward the near-critical edge, so the pre-threshold curvature is visible).
STDOUT prints ONLY: header "<n> <t>", then "<b> <h> <Hmax> <Tfail>", then n
rows "<Ta> <T>".  The hidden prefactor A, the critical ambient Ta_crit, and
the seeds are NEVER printed -- they live only inside gen.py / verify.py.
"""
import sys, math, random

# ---- fixed design constants (mirrored byte-for-byte in verify.py) ----
B_LO, B_HI = 0.04, 0.14
H_LO, H_HI = 2.5, 7.0
HMAX_CAP_FRAC = 0.62          # elbow width Hmax/h <= HMAX_CAP_FRAC/b < 1/b (tangency width)
EW_LO, EW_HI = 3.0, 8.0       # elbow width Hmax/h drawn directly -> healthy dT scale always
TACRIT_LO, TACRIT_HI = 28.0, 55.0
TRAIN_SPAN = 20.0             # training ambient window width, ending at the gap edge
GAP_LO, GAP_HI = 1.5, 4.0     # gap between highest training Ta and true Ta_crit
TFAIL_MARGIN_LO, TFAIL_MARGIN_HI = 20.0, 45.0
NOISE_TRAIN_FRAC = 0.10       # train noise sigma, as a fraction of the instance's elbow width
N_TRAIN = 46


def params(t):
    """Hidden thermal-runaway law for this test id (identical in verify.py)."""
    rng = random.Random(4058000 + t * 7919003)
    b = rng.uniform(B_LO, B_HI)
    h = rng.uniform(H_LO, H_HI)
    ew_cap = HMAX_CAP_FRAC / b               # cap on Hmax/h so the capacity limit
    ew_hi = min(EW_HI, ew_cap)               # binds strictly before Semenov tangency
    ew_lo = min(EW_LO, 0.8 * ew_hi)
    elbow_width = rng.uniform(ew_lo, ew_hi)  # = Hmax / h, drawn directly
    Hmax = elbow_width * h
    Ta_crit = rng.uniform(TACRIT_LO, TACRIT_HI)
    elbow = Ta_crit + elbow_width
    A = Hmax * math.exp(-b * elbow)
    margin = rng.uniform(TFAIL_MARGIN_LO, TFAIL_MARGIN_HI)
    Tfail = elbow + margin
    gap = rng.uniform(GAP_LO, GAP_HI)
    noise_train = NOISE_TRAIN_FRAC * elbow_width
    return b, h, Hmax, A, Ta_crit, Tfail, gap, noise_train


def true_T(Ta, b, h, Hmax, A, Tfail):
    """Steady-state temperature at ambient Ta, or Tfail if none exists."""
    elbow = Ta + Hmax / h

    def phi(T):
        return h * (T - Ta) - A * math.exp(min(700.0, b * T))

    if phi(elbow) < 0.0:
        return Tfail
    lo, hi = Ta, elbow
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if phi(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gen_train(t):
    b, h, Hmax, A, Ta_crit, Tfail, gap, noise_train = params(t)
    rng = random.Random(2231000 + t * 101)
    Ta_hi = Ta_crit - gap
    Ta_lo = Ta_hi - TRAIN_SPAN
    rows = []
    for _ in range(N_TRAIN):
        u = rng.random() ** 0.6          # bias toward the near-critical edge
        Ta = Ta_lo + (Ta_hi - Ta_lo) * u
        Ttrue = true_T(Ta, b, h, Hmax, A, Tfail)
        Tobs = Ttrue + rng.gauss(0.0, noise_train)
        rows.append((Ta, Tobs))
    rows.sort()
    return rows, (b, h, Hmax, A, Ta_crit, Tfail, Ta_lo, Ta_hi)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows, hidden = gen_train(t)
    b, h, Hmax, A, Ta_crit, Tfail, Ta_lo, Ta_hi = hidden
    out = ["%d %d" % (len(rows), t)]
    out.append("%.8g %.8g %.8g %.8g" % (b, h, Hmax, Tfail))
    for Ta, T in rows:
        out.append("%.8g %.8g" % (Ta, T))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
