import sys, random

# gen.py <testId>  -> prints ONE instance of the damper-seating-chart problem.
# All randomness is seeded from the testId only (fully deterministic).
#
# Format written to stdout:
#   line 1 : n K_total T_max
#   line 2 : c_unit beta
#   next n : m_j st_j     (mass of floor j ; story stiffness between floor j and j-1)

def main():
    i = int(sys.argv[1])
    rng = random.Random(52900 + 101 * i)

    if i <= 3:
        n = 12
    elif i <= 6:
        n = 24
    elif i <= 8:
        n = 40
    else:
        n = 64

    T_max = 4
    # budget factor cycles: 0.7 (few dampers -> placement decisive / trap regime),
    #                       1.2 (moderate), 1.8 (crowded, forced stacking).
    factor = [0.7, 1.2, 1.8][(i - 1) % 3]
    K_total = min(2 * n, n * T_max, max(3, round(factor * n)))

    m = [round(rng.uniform(0.6, 1.8), 4) for _ in range(n)]
    st = [round(rng.uniform(0.6, 1.8), 4) for _ in range(n)]

    # every third instance carries a heavy mass near the top (a resonant crown),
    # sharpening the fundamental antinode that lures the "damp the biggest sway" trap.
    if i % 3 == 0:
        for j in range(max(1, n - 3), n):
            m[j] = round(m[j] * rng.uniform(2.0, 3.5), 4)

    c_unit = 0.8   # viscous strength added per damper unit
    beta = 0.02    # stiffness-proportional material damping (naturally quiets fast modes)

    out = ["%d %d %d" % (n, K_total, T_max), "%.4f %.4f" % (c_unit, beta)]
    for j in range(n):
        out.append("%.4f %.4f" % (m[j], st[j]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
