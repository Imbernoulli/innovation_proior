# TIER: trivial
"""Reproduces the checker's own internal baseline exactly: 100% of the solvent
with the highest native anodic-stability threshold, zero additive. Safe but
slow -- ignores the conductivity/viscosity trade-off entirely."""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it))
    for _ in range(4):
        next(it)  # A_max V_target cov_target Kconst
    solv = []
    for _ in range(N):
        eta = float(next(it)); kappa = float(next(it)); thr = float(next(it))
        solv.append((eta, kappa, thr))

    i_star = max(range(N), key=lambda i: (solv[i][2], -i))
    x = [0.0] * N
    x[i_star] = 1.0
    a = [0.0] * M

    print(" ".join(f"{v:.6f}" for v in x))
    print(" ".join(f"{v:.6f}" for v in a))


if __name__ == "__main__":
    main()
