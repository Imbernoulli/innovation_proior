#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy wind-tunnel drag notebook to stdout.

A hidden drag law relates the drag force F on a bluff body to four measured
quantities, each carrying declared physical units:

    rho  fluid density        [kg m^-3]        (M L^-3)
    V    free-stream speed     [m s^-1]         (L T^-1)
    D    body length scale     [m]              (L)
    mu   dynamic viscosity     [kg m^-1 s^-1]   (M L^-1 T^-1)
    F    drag force            [kg m s^-2 = N]  (M L T^-2)   <- the target

The TRAIN notebook is what the solver SEES: a single wind-tunnel campaign in
ONE facility with ONE working fluid.  Consequently the campaign sweeps the
free-stream speed V over a wide range and the body length D over a moderate
range, but the fluid density rho and viscosity mu barely move (same air, same
tunnel) -- so their influence is NOT identifiable from the notebook by curve
fitting alone.  The HELD-OUT grading grid (regenerated ONLY inside the grader)
places all four quantities far outside these training ranges -- a different
fluid, a different scale -- so a fit that guessed the rho / mu behaviour from
the flat training columns extrapolates wildly, while the dimensionally forced
law does not.

STDOUT prints ONLY: header "<n> <test_id>" then n rows "<rho> <V> <D> <mu> <F>".
The hidden law, its coefficients, and the seeds are NEVER printed.
"""
import sys, math, random

# ---- fixed design constants (mirrored byte-for-byte in verify.py) ----
NARROW   = math.log(1.03)     # rho, mu jitter half-width (fluid nearly fixed)
DSWEEP   = math.log(3.0)      # body length swept moderately
VSWEEP   = math.log(8.0)      # free-stream speed swept widely (primary axis)
NOISE_TRAIN = 0.05            # multiplicative lognormal sensor noise (train)
N_TRAIN  = 200


def params(t):
    """Hidden drag law for this test id (identical in gen.py and verify.py)."""
    rng = random.Random(700000 + t * 9176111)
    a0 = rng.uniform(0.35, 1.10)     # asymptotic drag coefficient
    a1 = rng.uniform(3.0, 9.0)       # Re^-0.5 correction
    a2 = rng.uniform(6.0, 22.0)      # Re^-1  correction
    rho0 = math.exp(rng.uniform(math.log(1.0),  math.log(1000.0)))
    V0   = math.exp(rng.uniform(math.log(5.0),  math.log(60.0)))
    D0   = math.exp(rng.uniform(math.log(0.02), math.log(3.0)))
    mu0  = math.exp(rng.uniform(math.log(1e-5), math.log(1e-2)))
    return a0, a1, a2, rho0, V0, D0, mu0


def true_F(rho, V, D, mu, a0, a1, a2):
    """F = Cd(Re) * rho * V^2 * D^2, with Re = rho V D / mu (dimensionless)."""
    Re = rho * V * D / mu
    Cd = a0 + a1 * Re ** (-0.5) + a2 * Re ** (-1.0)
    return Cd * rho * V * V * D * D


def gen_train(t):
    a0, a1, a2, rho0, V0, D0, mu0 = params(t)
    rng = random.Random(111 + t * 13)
    rows = []
    for _ in range(N_TRAIN):
        V   = V0   * math.exp(rng.uniform(-VSWEEP, VSWEEP))
        D   = D0   * math.exp(rng.uniform(-DSWEEP, DSWEEP))
        rho = rho0 * math.exp(rng.uniform(-NARROW, NARROW))
        mu  = mu0  * math.exp(rng.uniform(-NARROW, NARROW))
        F   = true_F(rho, V, D, mu, a0, a1, a2) * math.exp(rng.gauss(0.0, NOISE_TRAIN))
        rows.append((rho, V, D, mu, F))
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = gen_train(t)
    out = ["%d %d" % (len(rows), t)]
    for rho, V, D, mu, F in rows:
        out.append("%.8g %.8g %.8g %.8g %.8g" % (rho, V, D, mu, F))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
