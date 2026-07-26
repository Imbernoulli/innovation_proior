# TIER: greedy
# The obvious "rank each converter by its own single-output efficiency, then use the
# best dedicated converter per demand" recipe: boiler exactly for steam (the turbine
# looks lossy in pure energy terms, since eps_p+eps_s < 1, so never route steam
# through it), the electric chiller for chill (its COP >> the absorption chiller's
# COP, so it "wins" a first-law efficiency ranking), and the grid for whatever
# power the electric chiller and the building still need. No cross-resource
# reasoning, no joint fuel accounting -- exactly minimal SIZING, but the wrong
# structural choice on cascade-favourable steps.
import sys


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
        b = float(S)
        x = 0.0
        z = 0.0
        e_chill = Ch / cop_elec
        e_grid = Pw + e_chill
        out.append("%.8f %.8f %.8f %.8f %.8f" % (b, x, z, e_chill, e_grid))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
