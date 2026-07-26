# TIER: strong
# The genuine insight: single-output "efficiency" ranking is the wrong lens. A unit of
# HP steam routed through the turbine is not "lossy" -- it is a JOINT fuel-to-two-demands
# purchase (electricity AND absorption-chill) paid for with ONE convex boiler increment,
# instead of TWO separate convex grid draws. So we stop asking "which converter is most
# efficient per demand" and instead ask, per timestep, "how much extra HP steam x should I
# push through the turbine, jointly accounting for every downstream demand it can touch,
# before the marginal boiler fuel outweighs the grid fuel it displaces?" That trade-off
# flips sign depending on the chill:power demand ratio (whether the turbine's LP-steam
# by-product has anywhere useful to go) -- exactly the cascade window the instance plants.
#
# For a FIXED x we can compute the cheapest feasible completion in closed form (spend the
# "free" turbine outputs first, top up any shortfall from the grid/electric chiller), so
# the whole per-timestep decision reduces to a 1-D search over x in [0, Cap_b - S]. We
# search with a coarse-to-fine grid (no unimodality assumed) -- a reformulation of the
# routing decision into one scalar knob, not "greedy plus more iterations."
import sys


def complete(a_b, c_b, a_g, c_g, eps_p, eps_s, cop_abs, cop_elec, S, Pw, Ch, x):
    b = S + x
    w = eps_p * x
    y = eps_s * x
    # spend LP steam on chill first (its only possible use)
    chill_from_lp = min(y, Ch / cop_abs if cop_abs > 0 else 0.0)
    z = chill_from_lp
    chill_rem = max(0.0, Ch - cop_abs * z)
    e_chill = chill_rem / cop_elec
    elec_needed = Pw + e_chill
    e_grid = max(0.0, elec_needed - w)
    F = a_b * b + c_b * b * b + a_g * e_grid + c_g * e_grid * e_grid
    return F, (b, x, z, e_chill, e_grid)


def best_for_step(a_b, c_b, a_g, c_g, eps_p, eps_s, cop_abs, cop_elec, S, Pw, Ch, Cap_b):
    lo, hi = 0.0, max(0.0, Cap_b - S)
    if hi <= 0.0:
        F, row = complete(a_b, c_b, a_g, c_g, eps_p, eps_s, cop_abs, cop_elec, S, Pw, Ch, 0.0)
        return row

    def ev(x):
        return complete(a_b, c_b, a_g, c_g, eps_p, eps_s, cop_abs, cop_elec, S, Pw, Ch, x)

    center, width = lo, hi - lo
    best_x, best_F, best_row = 0.0, None, None
    NPTS = 40
    for _pass in range(4):
        a = max(lo, center - width / 2.0)
        b_ = min(hi, center + width / 2.0)
        if _pass == 0:
            a, b_ = lo, hi
        for k in range(NPTS + 1):
            x = a + (b_ - a) * k / NPTS
            F, row = ev(x)
            if best_F is None or F < best_F:
                best_F, best_x, best_row = F, x, row
        center = best_x
        width = (b_ - a) / (NPTS / 4.0) + 1e-9
    return best_row


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
        row = best_for_step(a_b, c_b, a_g, c_g, eps_p, eps_s, cop_abs, cop_elec,
                             float(S), float(Pw), float(Ch), Cap_b)
        out.append("%.8f %.8f %.8f %.8f %.8f" % row)
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
