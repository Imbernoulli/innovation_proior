#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE battery-electrolyte-formulate instance to stdout.
Deterministic: seeded only by testId. Difficulty ladder small -> large/adversarial.

Story: formulate a liquid electrolyte from a library of N candidate solvents that
must be blended into volume fractions summing to 1, plus a library of M candidate
sacrificial SEI-forming additives dosed (in volume fraction) up to a small shared
budget A_max. Each solvent has a viscosity (eta), an intrinsic conductivity
coefficient (kappa), and a native anodic stability limit (thr) -- the voltage the
solvent alone survives before it decomposes at the anode. Each additive has an
SEI-forming strength (p) per unit loading (useful only up to its own coverage cap
-- excess loading buys no more protection but still costs viscosity/conductivity),
a viscosity penalty and a conductivity-dilution penalty per unit loading.

Mechanism composition:
  - conductivity-viscosity-tradeoff: the scored quantity IS a ratio of blended
    conductivity to blended viscosity (a Walden-rule-style quotient) -- solvents
    with the best raw conductivity are exactly the ones with the worst native
    stability window (planted anti-correlation), so optimizing the ratio alone
    walks straight into...
  - electrochemical-window: a hard pass/fail gate. If the blend's weakest-used
    solvent's native threshold is below the per-instance target window V_target,
    conductivity is worthless (F=0 for that case) UNLESS...
  - sei-forming-additive: enough sacrificial additive coverage is dosed to
    substitute for native solvent stability, decoupling the window requirement
    from the bulk solvent choice.

Trap (>=3 of 10, here 5 of 10, ids in TRAP_IDS): V_target is placed strictly
above the best-conductivity ("fast") solvent's native threshold but at or below
the safest solvent's threshold, so (a) a pure conductivity/viscosity optimizer
that ignores the window collapses to F=0, while (b) reachable additive combos
exist (within cap_j and the shared A_max budget) that push coverage over
cov_target -- so a solver that spots the decoupling trick keeps most of the fast
solvent's conductivity edge instead of retreating to a slow, native-safe solvent.
"""
import sys
import random

TRAP_IDS = {2, 4, 6, 8, 10}
SIZES = {1: (4, 2), 2: (5, 2), 3: (5, 3), 4: (6, 3), 5: (6, 3),
          6: (6, 3), 7: (7, 3), 8: (7, 4), 9: (7, 4), 10: (8, 4)}


def build(tid):
    rng = random.Random(31000 + 131 * tid)
    N, M = SIZES[tid]
    is_trap = tid in TRAP_IDS

    A_max = 0.12
    Kconst = 1.0

    solv = []
    for _ in range(N):
        eta = round(rng.uniform(1.5, 3.2), 3)
        kappa = round(rng.uniform(2.5, 4.5), 3)
        thr = round(rng.uniform(4.2, 5.4), 3)
        solv.append((eta, kappa, thr))

    # plant "fast" solvent at index 0: low viscosity, high conductivity coefficient,
    # but the WEAKEST native stability window in the library.
    fast_eta = round(rng.uniform(1.0, 1.4), 3)
    fast_kappa = round(rng.uniform(5.0, 6.5), 3)
    fast_thr = round(rng.uniform(3.6, 4.0), 3)
    solv[0] = (fast_eta, fast_kappa, fast_thr)

    # plant "safe" solvent at index 1 (if it exists): higher viscosity, modest
    # conductivity, but the library's strongest native window.
    safe_eta = round(rng.uniform(2.6, 3.4), 3)
    safe_kappa = round(rng.uniform(3.0, 4.2), 3)
    safe_thr = round(rng.uniform(5.0, 5.6), 3)
    if N > 1:
        solv[1] = (safe_eta, safe_kappa, safe_thr)

    max_thr = max(s[2] for s in solv)

    if is_trap:
        v_target = round(fast_thr + rng.uniform(0.25, 0.55), 3)
        v_target = min(v_target, max_thr - 0.05)
    else:
        v_target = round(fast_thr - rng.uniform(0.05, 0.30), 3)
    v_target = max(v_target, 0.5)

    add = []
    for _ in range(M):
        p = round(rng.uniform(3.0, 9.0), 3)
        etapen = round(rng.uniform(1.0, 3.0), 3)
        kappapen = round(rng.uniform(0.3, 1.2), 3)
        cap = round(rng.uniform(0.02, 0.06), 4)
        add.append((p, etapen, kappapen, cap))

    best_single_cov = max(p * cap for (p, etapen, kappapen, cap) in add)
    cov_target = round(rng.uniform(0.45, 0.85) * best_single_cov, 4)
    cov_target = max(cov_target, 0.01)

    lines = [f"{N} {M}",
             f"{A_max} {v_target} {cov_target} {Kconst}"]
    for (eta, kappa, thr) in solv:
        lines.append(f"{eta} {kappa} {thr}")
    for (p, etapen, kappapen, cap) in add:
        lines.append(f"{p} {etapen} {kappapen} {cap}")
    return "\n".join(lines) + "\n"


def main():
    tid = int(sys.argv[1])
    sys.stdout.write(build(tid))


if __name__ == "__main__":
    main()
