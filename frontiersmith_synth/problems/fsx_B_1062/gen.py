import sys, random

# Relay Throughput Spacing -- difficulty ladder, seeded ONLY by testId.
# Each test places m source/destination pairs along roughly parallel "lanes"
# (so different pairs' relay hops can interfere with each other) with
# DELIBERATELY HETEROGENEOUS lane lengths (short/easy pairs mixed with
# long/hard pairs) and a shared relay budget R.
#
# LADDER: m, R, lane gap (interference coupling), per-pair lane lengths,
# transmit power P, path-loss exponent alpha, noise floor N0.
LADDER = [
    # m, R,  gap,  lengths,                                   P,    alpha, N0
    (2,  2, 25.0, [15, 25],                                    12.0, 2.00, 0.20),
    (3,  4, 20.0, [12, 20, 30],                                12.0, 2.00, 0.20),
    (4,  8, 14.0, [10, 18, 26, 34],                            12.0, 2.00, 0.20),
    (4, 10, 12.0, [10, 20, 30, 40],                            11.0, 2.05, 0.20),
    (5, 14, 10.0, [8, 16, 24, 32, 40],                         11.0, 2.05, 0.18),
    (6, 20,  8.0, [8, 14, 20, 26, 32, 40],                     10.0, 2.10, 0.18),
    (6, 26,  7.0, [6, 12, 20, 28, 36, 46],                     10.0, 2.10, 0.15),
    (8, 28,  6.0, [6, 10, 16, 22, 28, 34, 40, 48],              9.0, 2.15, 0.15),
    (8, 30,  5.5, [6, 10, 14, 20, 26, 32, 40, 50],              8.0, 2.15, 0.12),
    (7, 24,  6.0, [6, 10, 14, 20, 28, 36, 48],                 10.0, 2.10, 0.15),
]

CX, CY = 30.0, 30.0
XMAX, YMAX = 60.0, 60.0
JITTER = 0.4


def main():
    testId = int(sys.argv[1])
    idx = min(max(testId, 1), len(LADDER)) - 1
    m, R, gap, lengths, P, alpha, N0 = LADDER[idx]
    rnd = random.Random(1000 + testId)  # seeded by testId only

    base_y = CY - (m - 1) * gap / 2.0
    pairs = []
    for i in range(m):
        y = base_y + i * gap + rnd.uniform(-JITTER, JITTER)
        L = lengths[i % len(lengths)]
        x0 = CX - L / 2.0 + rnd.uniform(-JITTER, JITTER)
        x1 = CX + L / 2.0 + rnd.uniform(-JITTER, JITTER)
        pairs.append((x0, y, x1, y))

    out = []
    out.append("%d %d %.4f %.4f %.4f %.4f %.4f" % (m, R, P, alpha, N0, XMAX, YMAX))
    for (sx, sy, dx, dy) in pairs:
        out.append("%.6f %.6f %.6f %.6f" % (sx, sy, dx, dy))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
