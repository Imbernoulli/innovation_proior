# TIER: greedy
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

    # Textbook steady-state reasoning: the fishery is healthy today (S0 well above
    # the Allee threshold A), so estimate today's "surplus production" g0 = growth(S0)
    # under the base recruitment rate, and harvest a constant fraction of it every
    # open week for the whole 5-year horizon. This is exactly the classic MSY /
    # replacement-yield heuristic -- it never looks at the printed low-recruitment
    # regime window, and it never re-checks whether the quota is still sustainable
    # once the stock or the regime has moved.
    g0 = r_base * S0 * (S0 / A - 1.0) * (1.0 - S0 / K)
    Q = max(0.0, 0.68 * g0)
    Q = min(Q, Qcap)

    lines = []
    for t in range(T):
        wk = t % 52
        lines.append("%.6f" % (0.0 if wk in closed_mod else Q))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
