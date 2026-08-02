# TIER: trivial
"""
Ignore wind-rose transport and the cluster structure entirely: calibrate ONE
scalar "everyone emits the same rate c" against the day-averaged, kernel-weighted
signal, and output c for every source. This is exactly the checker's own trivial
baseline construction, just estimated from the data instead of known outright.
"""
import sys, math


def kernel(sx, sy, wd, ws, SIGMA_MAX, ALPHA, L0):
    r = math.hypot(sx, sy)
    brg = math.degrees(math.atan2(-sy, -sx)) % 360.0
    d = abs(wd - brg) % 360.0
    delta = d if d <= 180.0 else 360.0 - d
    sigma = SIGMA_MAX / (1.0 + ALPHA * ws)
    return math.exp(-(delta * delta) / (2.0 * sigma * sigma)) * math.exp(-r / L0)


def dilution(ws, BETA):
    return 1.0 / (1.0 + BETA * ws)


def background(day_id, A0, A1, P):
    return A0 + A1 * math.sin(2.0 * math.pi * day_id / P)


def main():
    data = sys.stdin.read().split()
    p = 0
    test_id = int(data[p]); p += 1
    K = int(data[p]); p += 1
    D = int(data[p]); p += 1
    A0 = float(data[p]); p += 1
    A1 = float(data[p]); p += 1
    P = int(float(data[p])); p += 1
    SIGMA_MAX = float(data[p]); p += 1
    ALPHA = float(data[p]); p += 1
    L0 = float(data[p]); p += 1
    BETA = float(data[p]); p += 1

    sources = []
    for _ in range(K):
        sx = float(data[p]); p += 1
        sy = float(data[p]); p += 1
        sources.append((sx, sy))

    days = []
    for _ in range(D):
        day_id = int(data[p]); p += 1
        wd = float(data[p]); p += 1
        ws = float(data[p]); p += 1
        conc = float(data[p]); p += 1
        days.append((day_id, wd, ws, conc))

    num = 0.0
    den = 0.0
    for (day_id, wd, ws, conc) in days:
        rowsum = sum(kernel(sx, sy, wd, ws, SIGMA_MAX, ALPHA, L0) for (sx, sy) in sources)
        w = dilution(ws, BETA) * rowsum
        target = conc - background(day_id, A0, A1, P)
        num += w * target
        den += w * w
    c = num / den if den > 1e-9 else 0.0
    c = max(0.0, c)

    print(" ".join("%.6f" % c for _ in range(K)))


if __name__ == "__main__":
    main()
