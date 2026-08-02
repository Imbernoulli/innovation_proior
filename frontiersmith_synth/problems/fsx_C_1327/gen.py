#!/usr/bin/env python3
"""gen.py <testId> -- print ONE anti-reflection-coating instance to stdout.

Instance: a budget of N_max layers, K candidate coating materials (refractive
index n_i + a per-material deposition-thickness tolerance tol_i), the air/glass
boundary conditions (n0, n_sub), the design wavelength lambda0, and an incidence
angle range [0, theta_max_deg]. The solver must design a <=N_max layer stack
that stays good across the WHOLE angle range and under each layer's own
thickness tolerance (checked by verify.py via an exact worst-case transfer-
matrix grid) -- not just at the single nominal (normal incidence, zero error)
design point.

Higher-index materials are deliberately given proportionally larger tolerances
(harder to deposit precisely in practice) -- a genuine index/tolerance
trade-off a nominal-only optimizer cannot see. Later testIds widen theta_max
and the tolerances so a razor-sharp nominal null (large angle+thickness
sensitivity) is punished hard by the worst-case grid -- the trap.

Everything is seeded by testId only -> bit-for-bit reproducible.
"""
import sys
import random

# Fixed pool of realistic dielectric coating indices (MgF2 .. TiO2-ish), ascending.
POOL = [1.38, 1.46, 1.62, 1.78, 1.90, 2.05, 2.20, 2.35]

# (N_max, K, theta_max_deg, tol_lo, tol_hi, lambda0, n_sub)
SPECS = [
    (2, 2,  0, 0.005, 0.010, 550, 1.52),
    (2, 2, 10, 0.010, 0.020, 560, 1.50),
    (3, 3, 25, 0.020, 0.035, 540, 1.55),
    (3, 3, 35, 0.030, 0.050, 530, 1.58),
    (4, 3, 45, 0.040, 0.060, 560, 1.60),
    (4, 4, 50, 0.050, 0.070, 520, 1.65),
    (4, 4, 55, 0.050, 0.080, 580, 1.70),
    (5, 4, 60, 0.060, 0.090, 500, 1.75),
    (5, 5, 62, 0.070, 0.100, 610, 1.80),
    (5, 5, 65, 0.080, 0.110, 550, 1.85),
]


def pick_materials(K, rng):
    """Pick K distinct indices from POOL, always spanning the extremes (for
    K>=2) so both a low- and a high-index option are on the table."""
    n = len(POOL)
    forced = set()
    if K >= 1:
        forced.add(0)
    if K >= 2:
        forced.add(n - 1)
    remaining = [i for i in range(n) if i not in forced]
    rng.shuffle(remaining)
    need = max(0, K - len(forced))
    idxs = sorted(list(forced) + remaining[:need])
    return [POOL[i] for i in idxs]


def main():
    tid = int(sys.argv[1])
    rng = random.Random(20260726 + 97001 * tid)

    N_max, K, theta_max, tol_lo, tol_hi, lam0, n_sub = SPECS[(tid - 1) % len(SPECS)]
    n0 = 1.0

    mats = pick_materials(K, rng)
    lo, hi = POOL[0], POOL[-1]
    tols = []
    for ni in mats:
        frac = (ni - lo) / (hi - lo)
        tol = tol_lo + (tol_hi - tol_lo) * frac
        tols.append(round(tol, 5))

    # Decouple "position in the material list" from "index rank" -- material
    # 1 (the checker's do-nothing baseline pick) must NOT be systematically
    # the lowest-index / lowest-tolerance (hence most robust) option.
    order = list(range(K))
    rng.shuffle(order)
    mats = [mats[i] for i in order]
    tols = [tols[i] for i in order]

    out = []
    out.append("%d %d" % (N_max, K))
    for ni, ti in zip(mats, tols):
        out.append("%.4f %.5f" % (ni, ti))
    out.append("%.4f %.4f %.2f" % (n0, n_sub, float(lam0)))
    out.append("%d" % theta_max)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
