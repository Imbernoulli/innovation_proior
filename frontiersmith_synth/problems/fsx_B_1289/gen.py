#!/usr/bin/env python3
"""gen.py <testId> -- emits one capex-stage-gate instance to stdout.

Instance format (plain tokens):
  line 1: M
  line 2: c_1 ... c_M            (integer module build costs)
  line 3: a_1 ... a_M            (module signal accuracy in [0.50,0.97])
  line 4: p VG VB sigma F r      (prior(Good), payoff-if-Good, payoff-if-Bad,
                                   salvage rate, per-stage mobilization overhead,
                                   per-stage discount factor)

Cases are hand-tuned per testId (deterministic, no RNG needed): 1-2 are small
sanity/trap cases, 3-4 are "favourable" cases with zero information content
(full commitment is genuinely optimal there), 5-10 are information-per-dollar
traps of growing size (up to M=7), including two cases with TWO separated
informative modules that must be isolated into two different thin stages.
"""
import sys

# tid -> (M, costs, acc, p, VG, VB, sigma, F, r)
CASES = {
    1:  (3, [3, 20, 20],
         [0.88, 0.50, 0.50],
         0.50, 180, -70, 0.15, 2, 0.96),
    2:  (4, [3, 22, 22, 22],
         [0.90, 0.50, 0.50, 0.50],
         0.55, 220, -90, 0.15, 1, 0.96),
    3:  (4, [10, 15, 20, 25],
         [0.50, 0.50, 0.50, 0.50],
         0.85, 260, -30, 0.55, 1, 0.95),
    4:  (5, [12, 18, 22, 15, 20],
         [0.50, 0.50, 0.50, 0.50, 0.50],
         0.85, 330, -30, 0.55, 1, 0.96),
    5:  (5, [18, 4, 22, 20, 18],
         [0.50, 0.90, 0.50, 0.50, 0.50],
         0.60, 260, -150, 0.15, 3, 0.96),
    6:  (5, [4, 40, 40, 40, 40],
         [0.92, 0.50, 0.50, 0.50, 0.50],
         0.70, 300, -100, 0.55, 2, 0.97),
    7:  (6, [4, 35, 35, 35, 35, 35],
         [0.90, 0.50, 0.50, 0.50, 0.50, 0.50],
         0.60, 400, -100, 0.55, 2, 0.96),
    8:  (6, [4, 30, 30, 5, 30, 30],
         [0.90, 0.50, 0.50, 0.88, 0.50, 0.50],
         0.65, 300, -140, 0.10, 3, 0.96),
    9:  (7, [5, 45, 45, 45, 45, 45, 45],
         [0.92, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
         0.75, 450, -130, 0.55, 3, 0.96),
    10: (7, [5, 38, 38, 6, 38, 38, 38],
         [0.92, 0.50, 0.50, 0.88, 0.50, 0.50, 0.50],
         0.60, 450, -120, 0.10, 1, 0.96),
}


def main():
    tid = int(sys.argv[1])
    tid = ((tid - 1) % 10) + 1
    M, costs, acc, p, VG, VB, sigma, F, r = CASES[tid]
    out = []
    out.append(str(M))
    out.append(" ".join(str(c) for c in costs))
    out.append(" ".join("%.4f" % a for a in acc))
    out.append("%.4f %.4f %.4f %.4f %.4f %.4f" % (p, VG, VB, sigma, F, r))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
