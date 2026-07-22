#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE phyllotaxis-schedule instance to stdout.

Instance format:
  line 1: N R alpha
  line 2: p_bulk trans_frac p_rim
  line 3: score_frac

N          number of primordia (1..N)
R          final disk radius
alpha      lateral-inhibition strength coefficient (0,1)
p_bulk     growth exponent for the bulk region (k <= K0, K0=round(trans_frac*N))
trans_frac fraction of primordia in the bulk regime (1.0 = no separate rim regime)
p_rim      growth exponent for the transition/rim region (k > K0)
score_frac fraction (by count, counted from the outermost/last primordium) whose
           Voronoi cells feed the packing-uniformity score
"""
import sys

# (N, R, alpha, p_bulk, trans_frac, p_rim, score_frac)
TESTS = {
    1:  (20, 100.0, 0.60, 0.50, 1.00, 0.50, 0.30),
    2:  (25, 100.0, 0.55, 0.50, 1.00, 0.50, 0.30),
    3:  (30, 100.0, 0.50, 0.50, 0.72, 0.32, 0.30),
    4:  (32, 100.0, 0.50, 0.50, 0.68, 3.00, 0.30),
    5:  (35, 100.0, 0.50, 0.45, 1.00, 0.45, 0.30),
    6:  (38, 100.0, 0.50, 0.50, 0.65, 0.30, 0.30),
    7:  (42, 100.0, 0.45, 0.50, 0.60, 0.28, 0.30),
    8:  (46, 100.0, 0.45, 0.55, 0.68, 3.20, 0.30),
    9:  (50, 100.0, 0.40, 0.50, 1.00, 0.50, 0.30),
    10: (55, 100.0, 0.40, 0.50, 0.75, 0.35, 0.30),
}


def main():
    tid = int(sys.argv[1])
    if tid not in TESTS:
        tid = ((tid - 1) % 10) + 1
    N, R, alpha, p_bulk, trans_frac, p_rim, score_frac = TESTS[tid]
    print(f"{N} {R:.6f} {alpha:.6f}")
    print(f"{p_bulk:.6f} {trans_frac:.6f} {p_rim:.6f}")
    print(f"{score_frac:.6f}")


if __name__ == "__main__":
    main()
