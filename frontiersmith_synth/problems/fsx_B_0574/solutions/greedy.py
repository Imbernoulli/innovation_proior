# TIER: greedy
# The obvious textbook approach: classic GEOMETRIC staging (equal mass ratio per
# stage, which maximises the loss-FREE Tsiolkovsky cascade) with a single fixed
# uniform engine count per stage.  It is completely blind to the loss table L, so
# on the spiked-L (trap) instances -- where geometric staging piles the most
# propellant onto exactly the high-loss bottom stage and lets it burn long behind
# only two engines -- it lands far from the loss-aware optimum.
import sys

UNIFORM_N = 2  # a fixed "sensible" engine count per stage


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    S = int(float(next(it)))
    P = float(next(it)); M_total = float(next(it)); kappa = float(next(it))
    m_e = float(next(it)); T = float(next(it)); v_e = float(next(it))
    g = float(next(it)); E_max = int(float(next(it)))
    L = [float(next(it)) for _ in range(S)]  # read but deliberately unused

    n = [min(UNIFORM_N, E_max)] * S
    avail = (M_total - P - sum(n) * m_e) / (1.0 + kappa)
    if avail <= 0:
        n = [1] * S
        avail = (M_total - P - sum(n) * m_e) / (1.0 + kappa)

    # geometric mass split: propellant weight R^(S-i), R = (M_total/P)^(1/S)
    R = (M_total / P) ** (1.0 / S)
    w = [R ** (S - i) for i in range(S)]
    sw = sum(w)
    p = [avail * wi / sw for wi in w]

    out = ["%d %.6f" % (n[i], p[i]) for i in range(S)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
