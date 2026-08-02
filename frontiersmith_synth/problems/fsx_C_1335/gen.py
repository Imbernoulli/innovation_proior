#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE nanoparticle-synthesis instance to stdout.

Instance format (all whitespace-separated tokens):
    T L S
    r0 v0
    theta_ripen ripening_rate
    thr_0 cap_0 gcoef_0         (repeated L times, one per temperature level)
    ...
    bind_0 p_0                  (repeated S times, one per surfactant option)
    ...
    C0 max_inject
    target disp_limit

Seeded ONLY by testId (fully deterministic, no wall-clock/GPU/network).

Construction notes (why this is a genuine burst-vs-continuous-nucleation instance,
not a random number soup):
  - There are L=4 fixed "heat levels". Level i has a nucleation threshold thr_i
    (monomer pool must exceed this before ANY new nuclei form), a per-step
    nucleation-rate CAP cap_i (kinetics limit how many nuclei a single step can
    spawn even given a huge monomer excess), and a growth-rate coefficient
    gcoef_i. Thresholds DECREASE and caps/gcoef INCREASE with level index, i.e.
    hotter = easier to nucleate, faster to grow, and more nuclei-per-step
    capacity -- exactly the two-edged trade-off a solver must reason about: a
    hot level is the only way to burn through a monomer excess in one step
    (true burst separation), but staying hot after the burst re-triggers
    nucleation on every subsequent top-up.
  - A per-testId "kinetics scale" and "growth scale" perturb the whole level
    table together (order-preserving), so no two tests share numerics, while
    keeping a matched trivial/greedy/strong reference construction reliable
    across the family.
  - Three surfactant options differ in bind rate (how fast surface coverage
    saturates) and throttle exponent; only one is well matched to the
    instance's target size within the step budget -- a mismatched pick either
    caps growth long before reaching the target or barely throttles at all.
  - "trap" tests (>=3 of 10) use a TIGHT dispersity window relative to the
    growth needed. A schedule that lets nucleation dribble across many steps
    (the natural result of "just crank the heat and feed evenly", i.e. the
    textbook first attempt) ends up with particles of many different ages and
    therefore many different final sizes -- most of them miss a tight window
    even though their *average* size might look reasonable. A true single-
    burst schedule keeps every particle the same age, so it is immune to this
    by construction.
"""
import random
import sys

THR_BASE = [70.0, 55.0, 42.0, 30.0]
CAP = [1, 2, 4, 7]
GCOEF_BASE = [0.35, 0.55, 0.85, 1.35]
V0_BASE = 5.0
R0 = 2.0
THETA_RIPEN_BASE = 24.0
RIPEN_RATE = 0.18
L = 4
S = 3
TRAP_TESTS = (2, 3, 5, 6, 8, 9)


def params(test_id):
    rng = random.Random(31337 * test_id + 137)
    T = 11 + (test_id - 1) % 3  # 11,12,13 cycling -- modest difficulty ladder in horizon

    kin_scale = rng.uniform(0.85, 1.20)
    g_scale = rng.uniform(0.88, 1.18)

    thr = [x * kin_scale for x in THR_BASE]
    v0 = V0_BASE * kin_scale
    gcoef = [x * g_scale for x in GCOEF_BASE]
    theta_ripen = THETA_RIPEN_BASE * kin_scale

    burst_need = thr[L - 1] + CAP[L - 1] * v0
    max_inject = burst_need * rng.uniform(1.05, 1.2)
    C0 = burst_need * rng.uniform(2.6, 3.0)

    trap = test_id in TRAP_TESTS
    target = R0 + rng.uniform(5.0, 7.0)
    disp_frac = rng.uniform(0.13, 0.17) if trap else rng.uniform(0.20, 0.26)
    disp_limit = (target - R0) * disp_frac

    binds = sorted([rng.uniform(0.04, 0.07), rng.uniform(0.09, 0.13), rng.uniform(0.18, 0.26)])
    surf = [(b, rng.uniform(1.5, 1.9)) for b in binds]

    return dict(T=T, L=L, S=S, thr=thr, cap=CAP, gcoef=gcoef, v0=v0, r0=R0,
                theta_ripen=theta_ripen, ripening_rate=RIPEN_RATE,
                max_inject=max_inject, C0=C0, target=target, disp_limit=disp_limit,
                surf=surf)


def main():
    test_id = int(sys.argv[1])
    p = params(test_id)
    out = []
    out.append(f"{p['T']} {p['L']} {p['S']}")
    out.append(f"{p['r0']:.6f} {p['v0']:.6f}")
    out.append(f"{p['theta_ripen']:.6f} {p['ripening_rate']:.6f}")
    for i in range(p['L']):
        out.append(f"{p['thr'][i]:.6f} {p['cap'][i]} {p['gcoef'][i]:.6f}")
    for i in range(p['S']):
        b, ex = p['surf'][i]
        out.append(f"{b:.6f} {ex:.6f}")
    out.append(f"{p['C0']:.6f} {p['max_inject']:.6f}")
    out.append(f"{p['target']:.6f} {p['disp_limit']:.6f}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
