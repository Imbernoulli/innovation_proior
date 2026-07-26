# TIER: trivial
# Fully-dedicated construction (boiler only for steam, electric chiller only for
# chill, grid only for power), each quantity generously oversized/unoptimized -- this
# is EXACTLY the checker's internal baseline B, so it scores ~0.1 by construction.
import sys

SAFETY = 2.1


def main():
    toks = sys.stdin.read().split()
    p = 0
    T = int(toks[p]); p += 1
    a_b = float(toks[p]); p += 1
    c_b = float(toks[p]); p += 1
    Cap_b = float(toks[p]); p += 1
    eps_p = float(toks[p]); p += 1
    eps_s = float(toks[p]); p += 1
    cop_abs = float(toks[p]); p += 1
    cop_elec = float(toks[p]); p += 1
    a_g = float(toks[p]); p += 1
    c_g = float(toks[p]); p += 1

    out = []
    for t in range(T):
        S = int(toks[p]); p += 1
        Pw = int(toks[p]); p += 1
        Ch = int(toks[p]); p += 1
        b = SAFETY * S
        x = 0.0
        z = 0.0
        e_chill = SAFETY * Ch / cop_elec
        e_grid = SAFETY * (Pw + e_chill)
        out.append("%.8f %.8f %.8f %.8f %.8f" % (b, x, z, e_chill, e_grid))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
