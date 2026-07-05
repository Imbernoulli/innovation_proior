#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE early-phase concentration sample to stdout.

Drug-trial pharmacokinetics.  After an intravenous bolus dose, the plasma
concentration of a compound follows a hidden two-compartment (biexponential)
disposition law

    C(t) = A*exp(-alpha*t) + B*exp(-beta*t)          (A,B,alpha,beta > 0, alpha > beta)

where the fast rate `alpha` is the distribution phase and the slow rate `beta`
is the terminal elimination phase.  During the assay window of a Phase-I trial
you can only draw blood in the EARLY sampling window t in [0.08, 3.0] hours,
and every assay reading carries multiplicative measurement error.  Each test id
is a DIFFERENT compound (different hidden A,alpha,B,beta).

The grader regenerates a hidden LATE clearance tail (t in [4, 24] h) that is
never printed here; your recovered law is judged by how well it extrapolates
into that terminal region -- i.e. whether you separated the slow elimination
rate `beta` from the early mixed signal rather than merely fitting the curve.

Difficulty ladder (testId 1..10): more assay noise + fewer draws.
STDOUT prints ONLY: a header "<n_draws> <test_id>" then n_draws data rows
"<t> <C>".  The hidden law and seeds are NOT printed.
"""
import sys, math, random

T_LO, T_HI = 0.08, 3.0          # early sampling window (hours)


def coeffs(t):
    rng = random.Random(90001 + t * 7919)
    A = rng.uniform(6.0, 12.0)      # distribution-phase amplitude
    alpha = rng.uniform(1.3, 2.8)   # fast distribution rate (HIDDEN)
    B = rng.uniform(1.5, 4.0)       # terminal amplitude
    beta = rng.uniform(0.12, 0.35)  # slow elimination rate (HIDDEN)
    return A, alpha, B, beta


def fval(tt, cf):
    A, alpha, B, beta = cf
    return A * math.exp(-alpha * tt) + B * math.exp(-beta * tt)


def gen_train(t):
    n = 60 - (t - 1) * 4            # 60 down to 24 draws
    cv = 0.06 + (t - 1) * 0.02      # log-normal assay CV, 6% up to 24%
    cf = coeffs(t)
    rng = random.Random(500 + t * 104729)
    lo, hi = math.log(T_LO), math.log(T_HI)
    rows = []
    for _ in range(n):
        tt = math.exp(rng.uniform(lo, hi))   # PK-style early-dense sampling
        c = fval(tt, cf) * math.exp(rng.gauss(0.0, cv))
        rows.append((tt, c))
    return rows, n


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows, n = gen_train(t)
    out = ["%d %d" % (n, t)]
    for tt, c in rows:
        out.append("%r %r" % (tt, c))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
