# TIER: greedy
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)

    def nx():
        return next(it)

    T = int(nx())
    n0 = int(nx()); a0 = float(nx()); w0 = int(nx()); K = int(nx())
    for _ in range(K):
        nx()
    m = int(nx()); d_lo = int(nx()); d_hi = int(nx())
    cost_split = float(nx()); cost_delay = float(nx()); value_frac = float(nx())
    Lc = int(nx())
    for _ in range(Lc):
        nx(); nx(); nx(); nx()
    c1 = float(nx()); c2 = float(nx()); p = float(nx())
    Vmax = int(nx()); Wmax = int(nx()); Amax = float(nx())

    # "stop last week's attack": threshold-fit tightly to the exact observed base
    # pattern (n0 transactions of amount a0 within window w0). Ignores the attacker's
    # adaptation economics (cost_split, cost_delay, value_frac, m, d_lo, d_hi) and
    # ignores the legitimate customer clusters entirely.
    V = max(0, min(Vmax, n0 - 1))
    W = max(1, min(Wmax, w0))
    A = max(0.0, min(Amax, a0 - 0.01))
    print("%d %d %.6f" % (V, W, A))


if __name__ == "__main__":
    main()
