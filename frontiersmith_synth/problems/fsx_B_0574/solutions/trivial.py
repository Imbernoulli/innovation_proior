# TIER: trivial
# BASE_N engines per stage, propellant split equally across stages.
# This reproduces the grader's internal baseline -> ratio ~0.1.
import sys

BASE_N = 1


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    S = int(float(next(it)))
    P = float(next(it)); M_total = float(next(it)); kappa = float(next(it))
    m_e = float(next(it)); next(it); next(it); next(it); next(it)  # T, v_e, g, E_max
    total_prop = (M_total - P - S * BASE_N * m_e) / (1.0 + kappa)
    if total_prop < 0:
        total_prop = 0.0
    pi = total_prop / S
    out = ["%d %.6f" % (BASE_N, pi) for _ in range(S)]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
