#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample of a hidden quench-cooling
law to stdout.

Story: a smith quenches many workpieces and logs, for each, a temperature
reading T and the one-tick temperature CHANGE dT that followed it.  The
logger's thermometer only ever reaches a modest range (the metal is already
cooling by the time the smith can point it).  The true cooling rate is the SUM
of two physical regimes:

    dT = -( a*(T-AMBIENT) + b*(T-AMBIENT)**3 )

a linear conduction/convection term and a cubic radiative term.  Over the
logged range the cubic term is a MINOR correction next to conduction and next
to the thermometer's own reading noise -- it barely moves any single point.
Only its small, SYSTEMATIC bias across many points betrays it.

STDOUT prints ONLY: a header "<n_train> <test_id> <ambient>", then n_train
rows "<T> <dT>".  The coefficients a, b and the noise seed are NEVER printed
-- the ground truth lives only inside verify.py (duplicated there, not
imported, per contract).
"""
import sys, random

AMBIENT = 20.0
TRAIN_LO, TRAIN_HI = 30.0, 320.0


def params(t):
    """Hidden per-test law. Lives in gen.py AND verify.py (kept in sync by hand,
    never imported) so no ground-truth module is shipped in the directory."""
    rng = random.Random(900191 + t * 7919)
    a = rng.uniform(0.020, 0.050)
    frac_hi = 0.35 - 0.025 * (t - 1)          # cubic's share of the linear term
    frac_lo = max(0.03, frac_hi - 0.09)       # AT THE TOP of the train range;
    cubic_frac = rng.uniform(frac_lo, frac_hi)  # shrinks (harder to see) as t grows
    td = TRAIN_HI - AMBIENT
    b = cubic_frac * a / (td * td)
    noise_mult = rng.uniform(2.0, 3.0)
    # Calibrate sigma against the cubic term's value AT THE TOP of the train
    # range (the single largest it ever gets there) so it stays below the
    # per-point reading noise EVERYWHERE in the log, not just on average.
    sigma = noise_mult * b * (td ** 3)
    n = 2000 + 1200 * (t - 1)
    return a, b, sigma, n


def true_delta(T, a, b):
    x = T - AMBIENT
    return -(a * x + b * x ** 3)


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    a, b, sigma, n = params(t)
    rng = random.Random(31337 + t * 104729)
    out = ["%d %d %.6f" % (n, t, AMBIENT)]
    for _ in range(n):
        T = rng.uniform(TRAIN_LO, TRAIN_HI)
        d = true_delta(T, a, b) + rng.gauss(0.0, sigma)
        out.append("%.6f %.6f" % (T, d))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
