import sys, random

# ---- fixed model constants (identical across all testIds) ----
RHO_C = 3150.0     # cement specific weight, kg/m3
RHO_W = 1000.0     # water density, kg/m3
AIR = 0.02         # entrained-air volume fraction
C_MIN, C_MAX = 290.0, 520.0
W_MIN, W_MAX = 95.0, 230.0
WC_MIN, WC_MAX = 0.15, 0.75
WR_MAX, P_HALF, P_MAX = 0.40, 0.012, 0.035
K1, K2, K3, K4 = 5.0, 55.0, 42.0, 10.0
A, B = 55.0, 70.0
VAGG_MIN = 0.60

# per-testId schedule: (num aggregate blends, plant-a-trap flag)
SCHEDULE = {
    1:  (3, False),
    2:  (3, False),
    3:  (3, True),
    4:  (4, False),
    5:  (4, True),
    6:  (4, False),
    7:  (5, True),
    8:  (5, False),
    9:  (5, True),
    10: (6, False),
}


def scr(w, c, p):
    vc = c / RHO_C
    vw = w / RHO_W
    vagg = 1.0 - AIR - vc - vw
    return K1 * (w / c) + K2 * (vc + vw) - K3 * vagg + K4 * p


def greedy_scr_threshold(w0_1):
    # SCR of the naive (blend-1, p=0, c=c_max) recipe: the exact boundary risk_limit at
    # which that specific recipe flips feasible/infeasible.
    return scr(w0_1, C_MAX, 0.0)


def build(test_id):
    k, is_trap = SCHEDULE[test_id]
    rng = random.Random(9130000 + test_id)
    w0_1 = round(rng.uniform(193.0, 207.0), 1)
    w0 = [w0_1]
    lo, hi = 122.0, 190.0
    vals = sorted((round(rng.uniform(lo, hi), 1) for _ in range(k - 1)), reverse=True)
    w0.extend(vals)
    thr = greedy_scr_threshold(w0_1)
    if is_trap:
        risk_limit = round(thr - 2.5 - rng.uniform(0.0, 1.5), 2)
    else:
        risk_limit = round(thr + 2.5 + rng.uniform(0.0, 1.5), 2)
    return k, w0, risk_limit


def main():
    test_id = int(sys.argv[1])
    k, w0, risk_limit = build(test_id)

    lines = []
    lines.append(str(k))
    lines.append(f"{RHO_C:.6f} {RHO_W:.6f} {AIR:.6f}")
    lines.append(f"{C_MIN:.6f} {C_MAX:.6f}")
    lines.append(f"{W_MIN:.6f} {W_MAX:.6f}")
    lines.append(f"{WC_MIN:.6f} {WC_MAX:.6f}")
    lines.append(f"{WR_MAX:.6f} {P_HALF:.6f} {P_MAX:.6f}")
    lines.append(f"{K1:.6f} {K2:.6f} {K3:.6f} {K4:.6f}")
    lines.append(f"{A:.6f} {B:.6f}")
    lines.append(f"{VAGG_MIN:.6f} {risk_limit:.6f}")
    for x in w0:
        lines.append(f"{x:.6f}")

    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
