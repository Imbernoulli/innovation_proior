# TIER: trivial
# Axis (x-coordinate) equal-count split into K contiguous bands.  This exactly
# reproduces the evaluator's weak reference partition, so on every instance the
# candidate ARI equals the baseline ARI and the normalised score is ~0.1.
import sys, json

inst = json.load(sys.stdin)
pts = inst["points"]
k = inst["k"]
n = len(pts)

order = sorted(range(n), key=lambda i: (pts[i][0], i))
labels = [0] * n
base = n // k
extra = n % k
idx = 0
for band in range(k):
    cnt = base + (1 if band < extra else 0)
    for _ in range(cnt):
        labels[order[idx]] = band
        idx += 1

print(json.dumps({"labels": labels}))
