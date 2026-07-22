#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE refraction-lattice-lensing instance to stdout.
Deterministic: all randomness is seeded ONLY from testId via random.Random(seed).
No wall-time, no GPU, no external state.
"""
import sys, random


def clampi(v, lo, hi):
    return max(lo, min(hi, v))


def main():
    t = int(sys.argv[1])
    rng = random.Random(20260 + 7 * t)

    # ---- grid scales with testId: small/easy -> large/adversarial ----
    W = 16 + 2 * t + rng.randint(-1, 1)
    H = 13 + 2 * t + 2 * rng.randint(0, 1)   # keep H odd-ish spread, never tiny
    W = clampi(W, 12, 40)
    H = clampi(H, 11, 40)
    LMAX = 6
    area = W * H
    K = int(round((0.24 + 0.06 * rng.random()) * area))
    K = clampi(K, 20, area * LMAX)

    ALPHA = round(0.04 + 0.02 * rng.random(), 6)     # index unit per level
    KAPPA = round(0.32 + 0.16 * rng.random(), 6)     # gradient-index coupling strength

    # chromatic-splitting: three colors, strictly increasing dispersion coefficient
    chroma_r = round(0.45 + 0.25 * rng.random(), 6)     # < 1.0
    chroma_g = 1.0                                      # reference color
    chroma_b = round(1.35 + 0.45 * rng.random(), 6)     # > 1.0
    CHROMA = [chroma_r, chroma_g, chroma_b]

    # entry point: all three color bundles enter together (a single beam that must split)
    entry_frac = 0.16 + 0.10 * rng.random()
    entry = clampi(int(round(entry_frac * (H - 1))), 1, H - 2)

    # targets: monotonically spread in the SAME direction, spacing correlated with
    # chromatic dispersion (this is the planted structure 'strong' must discover:
    # differential chromatic bending, not a single symmetric focus, is what reaches
    # all three simultaneously).
    dr_frac = 0.08 + 0.06 * rng.random()
    dg_frac = 0.34 + 0.10 * rng.random()
    db_frac = 0.66 + 0.14 * rng.random()
    Tr = clampi(entry + int(round(dr_frac * (H - 1))), 0, H - 1)
    Tg = clampi(entry + int(round(dg_frac * (H - 1))), 0, H - 1)
    Tb = clampi(entry + int(round(db_frac * (H - 1))), 0, H - 1)
    # enforce strict separation (trap needs 3 genuinely distinct targets)
    if Tg <= Tr:
        Tg = min(H - 1, Tr + 1)
    if Tb <= Tg:
        Tb = min(H - 1, Tg + 1)
    T = [Tr, Tg, Tb]

    R = 3  # rays per color bundle
    offsets = [-1, 0, 1]
    rays = []
    for c in range(3):
        row = [clampi(entry + o, 0, H - 1) for o in offsets]
        rays.append(row)

    out = []
    out.append(f"{W} {H}")
    out.append(f"{LMAX} {K}")
    out.append(f"{ALPHA:.6f} {KAPPA:.6f}")
    out.append(f"{CHROMA[0]:.6f} {CHROMA[1]:.6f} {CHROMA[2]:.6f}")
    out.append(f"{T[0]} {T[1]} {T[2]}")
    out.append(f"{R}")
    for c in range(3):
        out.append(" ".join(str(v) for v in rays[c]))
    print("\n".join(out))


if __name__ == "__main__":
    main()
