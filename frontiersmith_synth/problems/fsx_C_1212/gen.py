#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN sample to stdout.

Traffic-metastable-forecast ("Free flow until one brake light").

A hidden freeway segment has a critical density RC (the peak of its
fundamental diagram) and a jam density RJ > RC.  Below RC the flow-density
relation is a classic single-valued free-flow branch.  At and above RC the
relation is METASTABLE and HYSTERETIC: the same density can sustain either
a near-capacity "still free" branch (if undisturbed) or a markedly lower
post-breakdown "discharge" branch (if a perturbation tipped it into a
moving jam) -- the textbook traffic capacity-drop phenomenon.

The solver only ever SEES data logged while density stayed comfortably
BELOW RC (genuinely single-valued, "free flow"). Each row also carries a
perturbation magnitude P (a local brake-event / disturbance size) and the
flow actually observed. At low density the flow barely reacts to P; the
SIZE of that reaction grows as density approaches RC -- a rising
perturbation-susceptibility that is the only clue, from safely sub-critical
data, to where/how sharply the metastable split will happen once density
climbs past RC. The held-out grading densities (regenerated only inside the
checker) lie in a HIGHER, non-overlapping range straddling RC and reaching
into the broken-down region -- never printed here.

STDOUT prints ONLY: a header "<n_train> <test_id>" then n_train rows
"<rho> <P> <q>". Hidden parameters and RNG seeds are NOT printed anywhere.
"""
import sys, random, math

EPS0 = 3.0     # fixed susceptibility-denominator floor (same for every test)
PMAX = 25.0    # fixed perturbation-magnitude range


def hidden_params(t):
    """Hidden road for this test id (lives in gen AND checker, never printed).
    RJ    -- jam density (flow -> 0)
    RC    -- critical density (fundamental-diagram peak); RC = frac_crit*RJ
    QM    -- capacity flow at RC
    S0    -- perturbation-susceptibility scale
    MU    -- metastable-branch decline rate past RC (small: near-capacity persists)
    DELTA -- capacity-drop fraction: discharge branch sits DELTA below QM at RC
    """
    rng = random.Random(812503 + t * 93179)
    RJ = rng.uniform(180.0, 260.0)
    frac_crit = rng.uniform(0.28, 0.38)
    RC = frac_crit * RJ
    QM = rng.uniform(1800.0, 2600.0)
    S0 = rng.uniform(150.0, 400.0)
    MU = rng.uniform(0.15, 0.35)
    DELTA = rng.uniform(0.30, 0.55)
    return RJ, RC, QM, S0, MU, DELTA


def schedule(t):
    """Difficulty ladder: visible sub-critical ceiling shrinks (less of the
    approach to RC is visible) and noise grows with t, so the faint rising
    susceptibility that reveals RC gets progressively harder to see."""
    u_train_max = [0.90, 0.87, 0.84, 0.81, 0.78, 0.75, 0.72, 0.69, 0.66, 0.63][t - 1]
    sigma_mult  = [0.02, 0.02, 0.025, 0.025, 0.03, 0.03, 0.035, 0.035, 0.04, 0.04][t - 1]
    n_train     = [70, 70, 65, 65, 60, 60, 55, 55, 50, 50][t - 1]
    return u_train_max, sigma_mult, n_train


def train_flow(rho, P, RC, QM, S0):
    """Free-flow branch (rho < RC): linear rise to capacity, perturbed by a
    susceptibility term that blows up (mildly, within the visible range) as
    rho approaches RC."""
    Fval = QM * rho / RC
    chi = S0 / (abs(rho - RC) + EPS0)
    return Fval - chi * P


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    RJ, RC, QM, S0, MU, DELTA = hidden_params(t)
    u_train_max, sigma_mult, n_train = schedule(t)
    rng = random.Random(2000011 + t * 7919)

    rows = []
    for _ in range(n_train):
        rho = RC * rng.uniform(0.05, u_train_max)
        P = rng.uniform(0.0, PMAX)
        q_clean = train_flow(rho, P, RC, QM, S0)
        q_clean = max(q_clean, 1.0)
        noise = rng.gauss(0.0, sigma_mult)
        q_obs = q_clean * math.exp(noise)
        rows.append((rho, P, q_obs))

    out = ["%d %d" % (n_train, t)]
    for rho, P, q in rows:
        out.append("%.6f %.6f %.6f" % (rho, P, q))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
