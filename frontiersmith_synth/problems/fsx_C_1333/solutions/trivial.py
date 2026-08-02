# TIER: trivial
# The checker's own reference recipe: blend 1 (the first/reference aggregate), no
# admixture, minimum cement, water = exactly what blend 1 needs for workability.
import sys


def main():
    tok = sys.stdin.read().split()
    it = iter(tok)
    K = int(next(it))
    rho_c = float(next(it)); rho_w = float(next(it)); air = float(next(it))
    c_min = float(next(it)); c_max = float(next(it))
    w_min = float(next(it)); w_max = float(next(it))
    wc_min = float(next(it)); wc_max = float(next(it))
    wr_max = float(next(it)); p_half = float(next(it)); p_max = float(next(it))
    k1 = float(next(it)); k2 = float(next(it)); k3 = float(next(it)); k4 = float(next(it))
    A = float(next(it)); B = float(next(it))
    vagg_min = float(next(it)); risk_limit = float(next(it))
    W0 = [float(next(it)) for _ in range(K)]

    j = 1
    c = c_min
    w = W0[0]
    p = 0.0
    print(j, c, w, p)


if __name__ == "__main__":
    main()
