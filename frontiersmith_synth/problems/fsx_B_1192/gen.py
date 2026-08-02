#!/usr/bin/env python3
"""gen.py <testId>  -- print ONE training log for the battery-knee-forecast problem.

Each row describes one cell cycled under fixed (temperature, depth-of-discharge)
conditions, summarized by two EARLY-WINDOW measurements -- a resistance-growth
rate and a capacity-fade rate, both estimated while the cell is still firmly in
its near-linear pre-knee regime -- plus a target cycle count at which we are
asked to forecast remaining capacity. Every training row's target cycle is,
BY CONSTRUCTION, still pre-knee for that cell (the knee has not happened yet
for anything in the log), matching real early-warning data collection.

Difficulty ladder (testId 1..N): larger testId => fewer rows and noisier
resistance/capacity-rate measurements, making the leading-indicator signal
harder to read out. Seeded via testId only.

STDOUT is DATA ROWS ONLY: six whitespace-separated floats
"x0 x1 x2 x3 x4 y" per line (temperature, depth-of-discharge, resistance-growth
rate, capacity-fade rate, target cycle, observed capacity fraction). The hidden
law, its coefficients, the sampling seed and the held-out region are NEVER
printed here -- the ground truth lives only inside the grader.
"""
import sys, math, random


# ---------------- private ground truth (grader keeps an identical copy) ----------------
N0 = 900.0          # baseline knee-onset cycle at zero cycling stress
KAPPA = 110.0        # cycles the knee moves earlier per unit of stress
ALPHA = 0.00035       # near-linear pre-knee fade rate (capacity fraction / cycle)
BETA = 0.006          # post-knee multiplicative collapse rate

R0 = 0.05             # resistance-growth-rate sensor offset
ETA = 0.55            # resistance-growth-rate sensitivity to stress

TRAIN_TEMP_LO, TRAIN_TEMP_HI = 0.0, 0.6
TRAIN_DOD_LO, TRAIN_DOD_HI = 0.0, 0.6
TRAIN_CYC_LO, TRAIN_CYC_HI = 60.0, 480.0   # stays pre-knee for ALL nominal-regime cells


def stress(temp, dod):
    return 1.0 + 1.3 * temp + 1.3 * dod + 1.6 * temp * dod


def n_knee(temp, dod):
    return N0 - KAPPA * stress(temp, dod)


def capacity_at(temp, dod, cyc):
    nk = n_knee(temp, dod)
    if cyc <= nk:
        return 1.0 - ALPHA * cyc
    y_knee = 1.0 - ALPHA * nk
    return y_knee * math.exp(-BETA * (cyc - nk))


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        return 1
    t = int(sys.argv[1])
    if t < 1:
        t = 1

    rng = random.Random(2000 + t)
    n_rows = 90 + 15 * t              # 105 .. 240 rows
    noise_R = 0.05 + 0.02 * t          # resistance-growth sensor noise grows with t
    noise_cap = 0.00006 + 0.00001 * t  # capacity-fade-rate sensor noise grows with t
    noise_y = 0.010 + 0.003 * t        # capacity measurement noise grows with t

    out = []
    for _ in range(n_rows):
        temp = rng.uniform(TRAIN_TEMP_LO, TRAIN_TEMP_HI)
        dod = rng.uniform(TRAIN_DOD_LO, TRAIN_DOD_HI)
        s = stress(temp, dod)

        x2 = R0 + ETA * s + rng.gauss(0.0, noise_R)          # resistance-growth rate (leading indicator)
        x3 = ALPHA + rng.gauss(0.0, noise_cap)                # capacity-fade rate (near-constant decoy)
        cyc = rng.uniform(TRAIN_CYC_LO, TRAIN_CYC_HI)          # always < this cell's own knee cycle

        y = capacity_at(temp, dod, cyc) + rng.gauss(0.0, noise_y)
        out.append("%.6f %.6f %.6f %.6f %.6f %.6f" % (temp, dod, x2, x3, cyc, y))

    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
