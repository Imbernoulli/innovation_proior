# TIER: greedy
import sys, json
inst = json.load(sys.stdin)
n = inst["n"]; budget = inst["budget"]
# coarse 3x3 uniform grid over the two DTLZ2 position variables; distance vars pinned to 0.5
# (so g=0 and every point sits on the front, but the batch is sparse).
lo = [0.0, 0.5, 1.0]
pts = []
for a in lo:
    for b in lo:
        pts.append([a, b] + [0.5] * (n - 2))
pts = pts[:budget]
print(json.dumps({"points": pts}))
