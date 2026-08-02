#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

Dose-response reverse engineering. For this test id there is a hidden compound
whose EFFICACY follows a saturating Hill/receptor-occupancy curve

    E(d) = Emax * d^n / (EC50^n + d^n)          (bounded above by Emax)

and whose TOXICITY follows some monotone, ACCELERATING, non-saturating curve
that keeps climbing (its exact shape is not told to the solver -- only that it
never levels off within the dosing range). All training measurements are taken
at LOW doses, well below where the interesting curvature and the toxicity
run-away become dramatic. STDOUT prints ONLY: a header "<n_train> <test_id>
<Dmax>" then n_train noisy rows "<dose> <efficacy> <toxicity>". The hidden
Hill/toxicity constants, Dmax's derivation, and the RNG seeds are NEVER
printed -- they live only in gen.py and verify.py (identically).
"""
import sys, random


def hidden_params(t):
    """Hidden dose-response law for this test id (also embedded in verify.py,
    NEVER printed). A different compound per test id."""
    rng = random.Random(90173 + t * 104729)
    Emax = rng.uniform(65.0, 95.0)          # efficacy ceiling (receptor pool is finite)
    n_hill = rng.uniform(1.4, 2.6)          # Hill (cooperativity) coefficient
    EC50 = rng.uniform(45.0, 85.0)          # dose at half-maximal efficacy (the saturation constant)
    Tbase = rng.uniform(1.5, 4.5)           # background toxicity at dose 0
    q = rng.uniform(1.5, 2.1)               # toxicity accelerates faster than linear
    Dmax = EC50 * (2.2 + 0.10 * (t - 1)) + rng.uniform(-5.0, 5.0)   # widest ethically allowed dose
    ratio_target = 1.3 + 0.07 * (t - 1) + rng.uniform(-0.05, 0.05)  # how far toxicity outruns efficacy by Dmax
    target_T_at_Dmax = Emax * ratio_target
    Tc = max(1e-6, (target_T_at_Dmax - Tbase) / (Dmax ** q))
    train_frac = 1.05 - 0.03 * (t - 1) + rng.uniform(-0.03, 0.03)   # training coverage, in units of EC50
    train_frac = max(0.55, train_frac)
    D_train_max = train_frac * EC50
    sigma_E = 0.8 + 0.05 * (t - 1)          # efficacy measurement noise sd
    sigma_T = 0.4 + 0.03 * (t - 1)          # toxicity measurement noise sd
    n_train = 16 + t
    return dict(Emax=Emax, n_hill=n_hill, EC50=EC50, Tbase=Tbase, q=q, Tc=Tc,
                Dmax=Dmax, D_train_max=D_train_max, sigma_E=sigma_E, sigma_T=sigma_T,
                n_train=n_train)


def E_true(d, p):
    dn = d ** p['n_hill']
    return p['Emax'] * dn / (p['EC50'] ** p['n_hill'] + dn)


def T_true(d, p):
    return p['Tbase'] + p['Tc'] * (d ** p['q'])


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    p = hidden_params(t)
    rng = random.Random(555001 + t * 7919)
    dmin = p['D_train_max'] * 0.04
    doses = sorted(dmin + (p['D_train_max'] - dmin) * rng.random() for _ in range(p['n_train']))

    out = ["%d %d %.6f" % (p['n_train'], t, p['Dmax'])]
    for d in doses:
        e = E_true(d, p) + rng.gauss(0.0, p['sigma_E'])
        tx = T_true(d, p) + rng.gauss(0.0, p['sigma_T'])
        out.append("%.6f %.6f %.6f" % (d, e, tx))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
