#!/usr/bin/env python3
# gen.py <testId>  -- prints ONE instance of the fast-pass pricing problem to stdout.
# Deterministic: all randomness seeded from testId only.
#
# Instance layout:
#   M N LAM T
#   for each ride j:  s_reg_j s_fast_j ref_j
#   for each visitor i: v_i K_i k_i r_1 ... r_{k_i}      (ride ids are 1-based)
#
# LAM = round(1000*lambda), lambda in [0,1].  T = number of equilibrium sweeps.
# ref_j = a deliberately-cheap "posted" reference price (the checker's baseline B is
#         the equilibrium objective at these ref prices; the trivial solution reproduces B).
#
# Design for a sharp scarcity trap:
#   * "collapse" rides have a LOW-capacity fast lane (s_fast just above s_reg) with large,
#     popular demand -> fast-lane savings collapse steeply as buyers flood in.
#   * time-values are BIMODAL: a thin tail of high-value "whales" and a large mass of
#     low-value "casuals", with a wide gap.  The static (empty-lane) demand curve makes
#     the casual mass look lucrative, but any positive crowding pushes realized savings
#     below what casuals need -> pricing for the masses sells almost nothing.
import sys, random, os

# tunable constants (env overrides for calibration sweeps; defaults are the locked values)
FW  = float(os.environ.get("FW",  "0.10"))   # whale fraction
VWL = int(os.environ.get("VWL", "108"))      # whale value lo
VWH = int(os.environ.get("VWH", "180"))      # whale value hi
VCL = int(os.environ.get("VCL", "10"))       # casual value lo
VCH = int(os.environ.get("VCH", "24"))       # casual value hi
RF  = float(os.environ.get("RF",  "0.20"))   # reference (baseline) price multiplier

def emit(testId):
    rnd = random.Random(1000003 * testId + 12289)

    if testId <= 2:
        M = 5 + testId; N = 500 * testId + 400; collapse_frac = 0.5
    elif testId <= 4:
        M = 9 + testId; N = 1600 + 500 * testId; collapse_frac = 0.7
    elif testId <= 7:
        M = 13 + testId; N = 3000 + 500 * testId; collapse_frac = 0.82
    else:
        M = 17 + testId; N = 4200 + 300 * testId; collapse_frac = 0.85
    M = min(M, 30); N = min(N, 7000)
    # small, revenue-dominant lambda (surplus is a minor tie-breaker / strategy knob)
    LAM = [0, 60, 120, 40, 150, 80, 130, 50, 160, 100][(testId - 1) % 10]
    T = 45

    rides = []                      # (s_reg, s_fast, is_collapse)
    for j in range(M):
        is_col = rnd.random() < collapse_frac
        s_reg = rnd.randint(4, 10)
        if is_col:
            s_fast = s_reg + rnd.randint(1, 2)        # steep collapse
        else:
            s_fast = s_reg * rnd.randint(6, 10)       # roomy lane
        rides.append([s_reg, s_fast, is_col])

    # bimodal visitors
    visitors = []
    for i in range(N):
        if rnd.random() < FW:
            v = rnd.randint(VWL, VWH)                 # whale
        else:
            v = rnd.randint(VCL, VCH)                 # casual
        # cardinality budget: mostly generous, a minority tight -> mild cross-ride coupling
        K = rnd.choice([2, 2, 3, 3, 4, 1])
        ksize = rnd.randint(2, min(6, M))
        pool = []
        for j in range(M):
            pool.extend([j] * (3 if rides[j][2] else 1))
        itin = set()
        while len(itin) < ksize:
            itin.add(rnd.choice(pool))
        visitors.append((v, K, sorted(itin)))

    demand = [0] * M
    for (v, K, itin) in visitors:
        for j in itin:
            demand[j] += 1

    # cheap posted reference prices -> flood the fast lanes (the "cheap passes" manager).
    # calibrated fraction of the LOW-value * empty-lane saving so B is a modest baseline.
    ref = []
    for j in range(M):
        n_j = max(1, demand[j]); S0 = n_j / rides[j][0]
        ref_j = max(1, int(round(RF * 14.0 * S0)))
        ref.append(ref_j)

    out = ["%d %d %d %d" % (M, N, LAM, T)]
    for j in range(M):
        out.append("%d %d %d" % (rides[j][0], rides[j][1], ref[j]))
    for (v, K, itin) in visitors:
        out.append("%d %d %d %s" % (v, K, len(itin), " ".join(str(j + 1) for j in itin)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    emit(tid)
