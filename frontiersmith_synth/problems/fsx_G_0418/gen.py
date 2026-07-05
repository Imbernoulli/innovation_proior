#!/usr/bin/env python3
"""Generator for fsx_G_0418 -- Arrhenius rate law for a catalytic packed-bed reactor.

`python3 gen.py <testId>` prints ONE train sample to stdout.

Only DATA ROWS are printed. The hidden ground-truth rate law, its coefficients and
the noise seed are NEVER emitted -- the solver must discover the Arrhenius functional
form from the (T, C, k) rows alone. The held-out EXTRAPOLATION split (hot, concentrated
operating points, far beyond the bench regime) lives only inside the checker.
"""
import sys, math, random

# ---- hidden ground-truth Arrhenius rate law (server-side; never printed) ----
#   k(T, C) = A * exp(-E / T) * C ** ORD
A_PRE, E_ACT, ORD, M_EXP = 2.0e5, 6500.0, 0.7, 1.0
def true_k(T, C):
    return A_PRE * (T ** M_EXP) * math.exp(-E_ACT / T) * (C ** ORD)


def main():
    tid = int(sys.argv[1])
    if tid < 1:
        tid = 1
    # difficulty ladder: higher testId -> more rows BUT more bench noise.
    M = 72 + 12 * tid
    s = 0.05 + 0.008 * tid          # multiplicative noise half-width on TRAIN
    rng = random.Random(4180 + tid)

    # bench regime: COLD (low T) and DILUTE (small C)
    C_CHOICES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

    rows = []
    for _ in range(M):
        T = rng.uniform(300.0, 440.0)          # cold band, kelvin
        C = rng.choice(C_CHOICES)              # dilute
        k = true_k(T, C) * (1.0 + rng.uniform(-s, s))
        rows.append((T, C, k))

    out = [str(M)]
    for T, C, k in rows:
        out.append("%.4f %.4f %.8e" % (T, C, k))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
