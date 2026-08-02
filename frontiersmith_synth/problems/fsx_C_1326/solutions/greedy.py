# TIER: greedy
"""The obvious recipe: maximize the conductivity/viscosity trade-off directly.
Picks the single solvent with the best kappa/eta ratio, uses 100% of it, and
never touches the additive library -- a pure conductivity-viscosity optimizer
that never models the electrochemical-window constraint at all. This is
individually rational (it IS the solvent that maximizes the ratio the
objective rewards) but walks straight into decomposition whenever that
solvent's native threshold sits below the instance's target window."""
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

    i_star = max(range(N), key=lambda i: solv[i][1] / solv[i][0])
    x = [0.0] * N
    x[i_star] = 1.0
    a = [0.0] * M

    print(" ".join(f"{v:.6f}" for v in x))
    print(" ".join(f"{v:.6f}" for v in a))


if __name__ == "__main__":
    main()
