# TIER: strong
# Sweep the number of rows R = 1..Rmax; for each build an offset lattice of N centers,
# then assign every disk the largest radius that stays inside the strip, non-overlapping
# with neighbours and clear of all pylons (min half-nearest-distance rule). Keep the R
# (and non-offset vs offset variant) with the largest total radius. Choosing R near
# sqrt(N*W/L) trades disk count against per-disk size -- the open-ended sweet spot the
# single/double-row heuristics miss, and it routes around the pylon obstacles.
import sys, math

t = sys.stdin.read().split()
N = int(t[0]); L = float(t[1]); W = float(t[2]); K = int(t[3]); RHO = float(t[4])
px = []; py = []
for j in range(K):
    px.append(float(t[5 + 2 * j])); py.append(float(t[6 + 2 * j]))


def build(R, offset):
    base = N // R
    extra = N % R
    centers = []
    for row in range(R):
        cnt = base + (1 if row < extra else 0)
        if cnt <= 0:
            continue
        yc = W * (row + 0.5) / R
        off = 0.5 if (offset and row % 2 == 1) else 0.0
        span = cnt + (0.5 if off else 0.0)
        for c in range(cnt):
            xc = (c + 0.5 + 0.5 * off) * L / span
            xc = min(max(xc, 1e-6), L - 1e-6)
            centers.append((xc, yc))
    return centers


def assign(centers):
    m = len(centers)
    rs = []
    tot = 0.0
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
        r = max(0.0, r - 1e-6)
        rs.append(r)
        tot += r
    return rs, tot


best_centers = None
best_rs = None
best_tot = -1.0
Rmax = min(9, N)
for R in range(1, Rmax + 1):
    for offset in (False, True):
        centers = build(R, offset)
        if not centers:
            continue
        rs, tot = assign(centers)
        if tot > best_tot:
            best_tot = tot; best_centers = centers; best_rs = rs

print(len(best_centers))
out = []
for (xc, yc), r in zip(best_centers, best_rs):
    out.append("%.10f %.10f %.10f" % (xc, yc, r))
sys.stdout.write("\n".join(out) + "\n")
