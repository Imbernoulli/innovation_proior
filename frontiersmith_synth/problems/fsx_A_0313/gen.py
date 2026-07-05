import sys
import random

# Solar-farm inverter clearance packing (format C, maximize sum of radii).
# A rectangular plot [0,W] x [0,H] contains M fixed circular keep-out zones
# (existing transformer pads / panel clusters). Place up to N circular inverter
# clearance zones, non-overlapping, inside the plot, and clear of every keep-out.
#
# Difficulty ladder (testId 1..10): N grows and the number of fixed keep-out
# obstacles grows from 0 (pure packing) to 9 (heavily obstructed / adversarial).
# Everything is deterministic in testId only.
LADDER_N = [8, 10, 12, 15, 18, 20, 24, 28, 32, 36]

i = int(sys.argv[1])
idx = min(max(i, 1), len(LADDER_N)) - 1
N = LADDER_N[idx]
W = round(8.0 + 0.6 * idx, 3)
H = round(6.0 + 0.4 * idx, 3)
M = idx  # 0 obstacles on the easiest case, up to 9 on the hardest

rng = random.Random(20260701 + 101 * idx)

obs = []
attempts = 0
while len(obs) < M and attempts < 5000:
    attempts += 1
    oR = round(rng.uniform(0.20, 0.55), 3)
    ox = round(rng.uniform(oR + 0.05, W - oR - 0.05), 3)
    oy = round(rng.uniform(oR + 0.05, H - oR - 0.05), 3)
    ok = True
    for (px, py, pr) in obs:
        dx = ox - px
        dy = oy - py
        if (dx * dx + dy * dy) ** 0.5 < oR + pr + 0.10:
            ok = False
            break
    if ok:
        obs.append((ox, oy, oR))

out = ["%d %s %s %d" % (N, ("%.3f" % W), ("%.3f" % H), len(obs))]
for (ox, oy, oR) in obs:
    out.append("%.3f %.3f %.3f" % (ox, oy, oR))
sys.stdout.write("\n".join(out) + "\n")
