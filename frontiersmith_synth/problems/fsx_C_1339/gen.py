import sys, math, random

# Difficulty ladder: season length T grows small -> larger.
LADDER_T = [8, 10, 12, 14, 16, 20, 22, 26, 30, 36]
# Trap testIds: force the K and Mg demand peaks to collide in time, so any schedule
# that co-applies K and Mg near the same day gets hit hard by ion antagonism.
TRAP_SET = {3, 5, 7, 9}


def gauss_curve(T, peak, width, mass, floor_frac=0.04):
    xs = []
    for t in range(1, T + 1):
        z = (t - peak) / max(width, 1e-6)
        xs.append(math.exp(-0.5 * z * z))
    s = sum(xs) if sum(xs) > 1e-12 else 1.0
    floor = floor_frac * mass / T
    return [ (x / s) * mass * (1.0 - floor_frac) + floor for x in xs ]


def main():
    testId = int(sys.argv[1])
    idx = min(max(testId, 1), len(LADDER_T)) - 1
    T = LADDER_T[idx]
    rng = random.Random(20000 + 97 * testId)

    P = max(3, round(T / 3.2))

    vN = round(rng.uniform(0.8, 1.6), 3)
    vK = round(rng.uniform(0.8, 1.6), 3)
    vMg = round(rng.uniform(0.8, 1.6), 3)

    retainN = round(rng.uniform(0.55, 0.72), 4)
    retainK = round(rng.uniform(0.75, 0.88), 4)
    retainMg = round(rng.uniform(0.85, 0.96), 4)

    if testId in TRAP_SET:
        kappa = round(rng.uniform(0.12, 0.30), 4)
    else:
        kappa = round(rng.uniform(0.30, 0.60), 4)

    MN = rng.uniform(80.0, 140.0)
    MK = rng.uniform(60.0, 110.0)
    MMg = rng.uniform(20.0, 45.0)

    widthN = rng.uniform(1.5, 3.0) * (T / 12.0)
    widthK = rng.uniform(1.5, 3.0) * (T / 12.0)
    widthMg = rng.uniform(1.3, 2.6) * (T / 12.0)

    peakN = rng.uniform(0.25, 0.45) * T
    peakK = rng.uniform(0.55, 0.78) * T

    if testId in TRAP_SET:
        peakMg = peakK + rng.uniform(-0.5, 0.5)
        widthMg = widthK * rng.uniform(0.9, 1.05)
    else:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        offset = (widthK + widthMg) * rng.uniform(1.8, 2.6)
        peakMg = peakK + sign * offset
        peakMg = min(max(peakMg, 1.0), float(T))

    DN = gauss_curve(T, peakN, widthN, MN)
    DK = gauss_curve(T, peakK, widthK, MK)
    DMg = gauss_curve(T, peakMg, widthMg, MMg)

    slack = rng.uniform(1.03, 1.15)
    BN = sum(DN) * slack
    BK = sum(DK) * slack
    BMg = sum(DMg) * slack

    out = []
    out.append("%d %d" % (T, P))
    out.append("%.4f %.4f %.4f" % (vN, vK, vMg))
    out.append("%.4f %.4f %.4f" % (retainN, retainK, retainMg))
    out.append("%.4f" % kappa)
    out.append("%.6f %.6f %.6f" % (BN, BK, BMg))
    for t in range(T):
        out.append("%.6f %.6f %.6f" % (DN[t], DK[t], DMg[t]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
