# TIER: greedy
# Two offset rows. Using the extra highway width lets each disk grow larger than a
# single row (fewer disks per unit length -> bigger radius), so total radius beats the
# trivial single-row baseline. Radii assigned by the always-feasible "min half nearest
# distance, capped by walls and pylons" rule.
import sys, math

t = sys.stdin.read().split()
N = int(t[0]); L = float(t[1]); W = float(t[2]); K = int(t[3]); RHO = float(t[4])
px = []; py = []
for j in range(K):
    px.append(float(t[5 + 2 * j])); py.append(float(t[6 + 2 * j]))

R = 2
base = N // R
extra = N % R
centers = []
for row in range(R):
    cnt = base + (1 if row < extra else 0)
    if cnt <= 0:
        continue
    yc = W * (row + 0.5) / R
    off = 0.5 if (row % 2 == 1) else 0.0
    for c in range(cnt):
        xc = (c + 0.5 + 0.5 * off) * L / (cnt + (0.5 if off else 0.0))
        xc = min(max(xc, 1e-6), L - 1e-6)
        centers.append((xc, yc))


def assign(centers):
    m = len(centers)
    rs = []
    for i in range(m):
        xi, yi = centers[i]
        r = min(xi, L - xi, yi, W - yi)
        for j in range(m):
            if j == i:
                continue
            xj, yj = centers[j]
            d = math.hypot(xi - xj, yi - yj)
            if d / 2.0 < r:
                r = d / 2.0
        for j in range(K):
            d = math.hypot(xi - px[j], yi - py[j]) - RHO
            if d < r:
                r = d
        rs.append(max(0.0, r - 1e-6))
    return rs


rs = assign(centers)
print(len(centers))
out = []
for (xc, yc), r in zip(centers, rs):
    out.append("%.10f %.10f %.10f" % (xc, yc, r))
sys.stdout.write("\n".join(out) + "\n")
