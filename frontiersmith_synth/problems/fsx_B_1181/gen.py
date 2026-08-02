#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE battery ECM-identification instance to stdout.

Theme: "Naming the internal circuit from a charge-discharge curve." A cell is
driven by a recorded current profile I(t) (a drive cycle mixing pulses of a
few different durations) and its terminal voltage V(t) is logged (with
measurement noise). The TRUE circuit is a series resistance R0, a small set
of parallel RC branches with hidden time constants, and a current-sign-driven
hysteresis voltage of hidden magnitude M. The solver must emit an ECM (R0, a
set of (R,C) branches, M) that predicts terminal voltage well -- INCLUDING on
a held-out drive cycle (regenerated only inside verify.py) whose current
profile mixes a genuinely different set of pulse durations.

Difficulty ladder / trap plant (testId 1..10):
  - "resolved" ids {1,2,3,7,8}: true circuit has 2 well-separated RC time
    constants (~15s, ~250s); the training drive cycle's two pulse-duration
    clusters sit right on top of them, so both are honestly resolvable.
  - "trap" ids {4,5,6,9,10}: true circuit has 3 RC branches, but two of them
    (~180s and ~230s, ratio 1.28x) are NEAR-DEGENERATE, and the training
    drive cycle's long-pulse cluster is a single NARROW band (190-210s) that
    cannot excite them differently -- their individual (R,C) split is not
    identifiable from this one cycle, only their combined effect is. A fit
    that claims 3+ branches anyway apportions the shared response between the
    two collinear branches however training noise happens to break the tie;
    that split is wrong and shows up once the held-out cycle's DIFFERENT
    pulse-duration mix (verify.py) asks the two branches to actually behave
    distinctly.
  N (profile length) grows with testId (small -> large instances).

STDOUT prints ONLY the recorded drive cycle (current + noisy voltage), the
OCV table, and generic (test-independent) parameter bounds -- never the true
R0 / branch values / M / noise seed / which regime this test id is.
"""
import sys
import math
import random

DT = 1.0
TAU_H = 90.0
CAPACITY_AH = 5.0
SOC_INIT = 0.85
KMAX = 8
R0_LO, R0_HI = 0.001, 1.0
R_LO, R_HI = 0.0001, 1.0
C_LO, C_HI = 1.0, 3_000_000.0
TAU_LO, TAU_HI = 0.5, 4000.0
M_LO, M_HI = -0.5, 0.5

# fixed OCV(soc) table, same cell chemistry for every test id
OCV_SOC_PTS = [round(0.05 * j, 2) for j in range(21)]  # 0.00 .. 1.00


def ocv_true(soc):
    return 3.00 + 1.20 * soc - 0.30 * soc * soc


# ---- HIDDEN ground truth (identical formulas duplicated in verify.py) ----
def gt_params(tid):
    trap = tid in (4, 5, 6, 9, 10)
    if not trap:
        taus = [15.0, 250.0]
        Rs = [0.028, 0.020]
    else:
        taus = [15.0, 180.0, 230.0]
        Rs = [0.028, 0.014, 0.014]
    R0 = 0.045 + 0.003 * (tid % 5)
    M_true = 0.012 + 0.0015 * tid
    sigma = 0.003
    return taus, Rs, R0, M_true, sigma, trap


# pulse-duration cluster ranges (seconds) per regime/role -- HIDDEN structure,
# only its numeric consequence (the I/V arrays) is ever printed.
PROFILES = {
    "resolved_train": [(10, 20), (220, 280)],
    "resolved_held": [(5, 10), (60, 100), (300, 400)],
    "trap_train": [(10, 20), (190, 210)],
    "trap_held": [(5, 15), (120, 160), (350, 450)],
}


def gen_profile(n, seed, clusters):
    rnd = random.Random(seed)
    I = []
    while len(I) < n:
        lo, hi = rnd.choice(clusters)
        dur = rnd.randint(lo, hi)
        amp = rnd.uniform(1.0, 4.0) * rnd.choice([-1, 1])
        I += [amp] * dur
    return I[:n]


def simulate(I, R0, taus, Rs, M, noise_sigma=0.0, noise_seed=0):
    n = len(I)
    K = len(taus)
    a = [math.exp(-DT / t) for t in taus]
    ah = math.exp(-DT / TAU_H)
    vb = [0.0] * K
    h = 0.0
    soc = SOC_INIT
    V = [0.0] * n
    rnd = random.Random(noise_seed) if noise_sigma > 0 else None
    for k in range(n):
        Ik = I[k]
        vt = ocv_true(soc) - R0 * Ik - sum(vb) - h
        if rnd is not None:
            vt += rnd.gauss(0.0, noise_sigma)
        V[k] = vt
        for i in range(K):
            vb[i] = a[i] * vb[i] + Rs[i] * (1 - a[i]) * Ik
        sgn = 0.0 if Ik == 0 else (1.0 if Ik > 0 else -1.0)
        h = ah * h + M * (1 - ah) * sgn
        soc -= Ik * DT / (3600.0 * CAPACITY_AH)
    return V


def main():
    if len(sys.argv) < 2:
        print("usage: gen.py <testId>", file=sys.stderr)
        sys.exit(1)
    tid = int(sys.argv[1])

    taus, Rs, R0, M_true, sigma, trap = gt_params(tid)
    n_train = 2200 + 200 * (tid - 1)  # 2200 .. 4000, small -> large ladder
    key = "trap_train" if trap else "resolved_train"
    I_train = gen_profile(n_train, seed=10_000 + tid, clusters=PROFILES[key])
    V_train = simulate(I_train, R0, taus, Rs, M_true, noise_sigma=sigma, noise_seed=20_000 + tid)

    out = []
    out.append(
        "%d %d %.1f %d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.1f %.2f %.4f"
        % (n_train, tid, DT, KMAX, R0_LO, R0_HI, R_LO, R_HI, C_LO, C_HI,
           TAU_LO, TAU_HI, M_LO, M_HI, TAU_H, CAPACITY_AH, SOC_INIT)
    )
    out.append(str(len(OCV_SOC_PTS)))
    for s in OCV_SOC_PTS:
        out.append("%.4f %.6f" % (s, ocv_true(s)))
    out.append(" ".join("%.4f" % x for x in I_train))
    out.append(" ".join("%.6f" % x for x in V_train))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
