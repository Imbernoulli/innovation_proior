# TIER: greedy
# Keep the coarse grid centres, but grow each clearance circle sequentially to
# the largest radius allowed by the plot walls, the keep-out obstacles, and the
# circles already fixed. Filling the interstitial slack beats the fixed-radius
# grid baseline.
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

centers = []
count = 0
for rr in range(gr):
    for cc in range(gc):
        if count >= N:
            break
        centers.append(((cc + 0.5) * cw, (rr + 0.5) * ch))
        count += 1

placed = []
for (cx, cy) in centers:
    rad = min(cx, W - cx, cy, H - cy)
    for (ox, oy, oR) in obs:
        rad = min(rad, math.hypot(cx - ox, cy - oy) - oR)
    for (px, py, pr) in placed:
        rad = min(rad, math.hypot(cx - px, cy - py) - pr)
    if rad > 0.0:
        placed.append((cx, cy, rad))

out = [str(len(placed))]
for (x, y, r) in placed:
    out.append("%.10f %.10f %.10f" % (x, y, r))
sys.stdout.write("\n".join(out) + "\n")
