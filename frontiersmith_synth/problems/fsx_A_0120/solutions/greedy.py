# TIER: greedy
# Global temperature bucketing: split the observed temperature range into B equal
# bands and label each cell by the band its ideal temperature falls in.  This
# slashes mismatch versus one zone, but it ignores geometry entirely -- noisy
# neighbours land in different bands, so it pays for a lot of walls and never
# reoptimizes the band count.
import sys, json

inst = json.load(sys.stdin)
H = inst["H"]
W = inst["W"]
flat = [t for row in inst["ideal"] for t in row]
mn = min(flat)
mx = max(flat)
B = 4
span = (mx - mn) if mx > mn else 1
labels = []
for t in flat:
    b = int((t - mn) * B / (span + 1e-9))
    if b >= B:
        b = B - 1
    if b < 0:
        b = 0
    labels.append(b)
print(json.dumps({"labels": labels}))
