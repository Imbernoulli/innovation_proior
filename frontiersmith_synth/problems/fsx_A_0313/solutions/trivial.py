# TIER: trivial
# Reproduces the checker's internal baseline exactly: a coarse row-major grid of
# small equal-ish clearance circles, each shrunk to respect the plot walls and
# the fixed keep-out obstacles. Total radius == B, so this scores ~0.1.
import sys
import math


def read_instance():
    toks = sys.stdin.read().split()
    p = 0
    N = int(toks[p]); p += 1
    W = float(toks[p]); p += 1
    H = float(toks[p]); p += 1
    M = int(toks[p]); p += 1
    obs = []
    for _ in range(M):
        ox = float(toks[p]); p += 1
        oy = float(toks[p]); p += 1
        oR = float(toks[p]); p += 1
        obs.append((ox, oy, oR))
    return N, W, H, obs


N, W, H, obs = read_instance()

gc = int(math.ceil(math.sqrt(N)))
gr = int(math.ceil(float(N) / gc))
cw = W / gc
ch = H / gr
frac = 0.30

circles = []
count = 0
for rr in range(gr):
    for cc in range(gc):
        if count >= N:
            break
        cx = (cc + 0.5) * cw
        cy = (rr + 0.5) * ch
        rad = frac * min(cw, ch)
        rad = min(rad, cx, W - cx, cy, H - cy)
        for (ox, oy, oR) in obs:
            d = math.hypot(cx - ox, cy - oy)
            rad = min(rad, d - oR)
        if rad > 0.0:
            circles.append((cx, cy, rad))
        count += 1

out = [str(len(circles))]
for (x, y, r) in circles:
    out.append("%.10f %.10f %.10f" % (x, y, r))
sys.stdout.write("\n".join(out) + "\n")
