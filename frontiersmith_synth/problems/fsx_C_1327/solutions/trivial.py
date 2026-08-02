# TIER: trivial
"""Reproduces the checker's own internal baseline: a single layer of the
FIRST provided material, at its nominal normal-incidence quarter-wave
thickness. No search, no worst-case reasoning at all -> should land right
on the checker's baseline B (Ratio ~ 0.1)."""
import sys

D_MAX = 1200.0


def main():
    toks = sys.stdin.read().split()
    p = 0
    N_max = int(toks[p]); p += 1
    K = int(toks[p]); p += 1
    materials = []
    for _ in range(K):
        ni = float(toks[p]); p += 1
        ti = float(toks[p]); p += 1
        materials.append((ni, ti))
    n0 = float(toks[p]); p += 1
    n_sub = float(toks[p]); p += 1
    lam0 = float(toks[p]); p += 1
    # theta_max unused by the trivial baseline construction

    ni = materials[0][0]
    d = lam0 / (4.0 * ni)
    d = min(max(d, 1e-3), D_MAX)

    print(1)
    print("%d %.6f" % (1, d))


if __name__ == "__main__":
    main()
