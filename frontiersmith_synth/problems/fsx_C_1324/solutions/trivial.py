# TIER: trivial
"""Wide-open, chemistry-free membrane: one pore family at the largest allowed
radius, no functionalization. This is EXACTLY the checker's own internal
baseline construction, so it reproduces B and scores Ratio ~0.1 on every
case."""
import sys


def main():
    toks = sys.stdin.read().split()
    (d_T, d_C, chi_T, chi_C, base_sol_T, base_sol_C,
     K_max, r_min, r_max, alpha_max, delta_coat, P_min) = toks[:12]
    r_max = float(r_max)
    print(1)
    print(0.0)
    print("%.6f %.6f" % (r_max, 1.0))


if __name__ == "__main__":
    main()
