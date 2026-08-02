# TIER: greedy
# The obvious "textbook" instinct: strength goes up as water/cement goes down and as
# cement goes up, so use the default (first-listed) aggregate, skip the admixture
# entirely (why bother with chemistry?), add just enough water to hit the workability
# target for that default aggregate, and push cement to the budget ceiling. This
# ignores that more cement (more paste) and a thirstier aggregate both drive up
# shrinkage-cracking risk -- so on several instances this recipe blows the shrinkage
# budget outright.
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
    w = W0[0]           # default aggregate, exactly the water it demands
    p = 0.0              # no admixture
    c = c_max            # more cement = more strength, so max it out
    print(j, c, w, p)


if __name__ == "__main__":
    main()
