# TIER: trivial
import sys


def main():
    d = sys.stdin.read().split()
    it = iter(d)
    T = int(next(it))
    K, A, Acol, gamma, p_base, Qcap = (float(next(it)) for _ in range(6))
    S0, r_base = (float(next(it)) for _ in range(2))
    dstart = int(next(it)); dend = int(next(it)); drought_mult = float(next(it))
    n_closed = int(next(it))
    closed_mod = set(int(next(it)) for _ in range(n_closed))
    n_season = int(next(it))
    price_season = [float(next(it)) for _ in range(n_season)]
    N = int(next(it))
    for _ in range(N):
        next(it); next(it)

    # naive flat constant harvest: a small fixed fraction of the initial stock,
    # applied every week except the mandatory legal closed weeks. Ignores the
    # depensation dynamics and the drought regime window entirely.
    Qb = min(0.012 * S0, Qcap)
    lines = []
    for t in range(T):
        wk = t % 52
        lines.append("%.6f" % (0.0 if wk in closed_mod else Qb))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
