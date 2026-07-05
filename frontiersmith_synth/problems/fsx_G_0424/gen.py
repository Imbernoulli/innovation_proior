#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE wind-tunnel training sample to stdout.

Aerodynamics / wind-tunnel drag calibration.  For one bluff-body model the
measured drag coefficient `Cd` is governed by a fixed but UNKNOWN multi-regime
closed-form law over

    Re  -- Reynolds number (flow speed x chord / kinematic viscosity)
    eps -- relative surface roughness of the model

Real drag laws span several regimes: a viscous 1/Re creeping-flow term, a
sqrt-Reynolds boundary-layer term, a bluff-body pressure-drag plateau, and a
high-Re "drag-crisis" drop where the boundary layer trips turbulent and the
wake narrows.  During a routine wind-tunnel campaign you can only run the tunnel
in its MID-Reynolds calibration band (Re ~ 10^1.5 .. 10^3.5), and every reading
carries balance/transducer noise.  The engineers need a compact analytic law
that still predicts drag in the HIGH-Reynolds region (Re up to 10^5), which the
tunnel cannot reach directly.

Each test id fixes a DIFFERENT hidden law (a different model geometry).  The
solver only sees noisy MID-Re measurements; the high-Re held-out split lives in
the grader and is never printed here.

STDOUT: a header "<n_train> <test_id>" then n_train rows "Re eps Cd".
The hidden law, its coefficients and seeds are NOT printed.
"""
import sys, math, random


def coeffs(t):
    rng = random.Random(1300007 + t * 49999)
    a = rng.uniform(18.0, 30.0)     # creeping-flow 1/Re amplitude
    b = rng.uniform(3.0, 6.0)       # boundary-layer 1/sqrt(Re) amplitude
    c = rng.uniform(0.35, 0.55)     # bluff-body pressure-drag plateau
    h = rng.uniform(0.20, 0.60)     # roughness lift of the plateau
    d = rng.uniform(0.15, 0.35)     # drag-crisis drop amplitude
    logRc = rng.uniform(3.70, 4.30)  # crisis onset (HIGH-Re, outside train)
    Rc = 10.0 ** logRc
    m = rng.uniform(2.0, 4.0)       # crisis sharpness
    g = rng.uniform(0.5, 1.5)       # roughness advances the crisis
    return (a, b, c, h, d, Rc, m, g)


def fval(Re, eps, cf):
    a, b, c, h, d, Rc, m, g = cf
    return (a / Re
            + b / math.sqrt(Re)
            + c + h * eps
            + d / (1.0 + (Re / (Rc * (1.0 + g * eps))) ** m))


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sigma = 0.02 + (t - 1) * 0.012
    n = 220 - (t - 1) * 16
    cf = coeffs(t)
    rng = random.Random(880023 + t * 100003)
    out = ["%d %d" % (n, t)]
    for _ in range(n):
        Re = 10.0 ** rng.uniform(1.5, 3.5)
        eps = rng.uniform(0.0, 0.05)
        Cd = fval(Re, eps, cf) + rng.gauss(0.0, sigma)
        out.append("%r %r %r" % (Re, eps, Cd))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
