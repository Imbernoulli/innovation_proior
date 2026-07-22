# TIER: greedy
# The obvious "maximize average detection" recipe: split the shared 1% false-
# positive allowance EQUALLY across every sensor (fp_cap / K each), threshold
# each sensor at its own quantile for that equal share, and OR-fuse them
# (weight 1 each, vote threshold 1 -- any single sensor firing flags the
# event). This is a single, common operating point applied uniformly: it
# never asks which family needs more of the allowance and which needs less,
# so it starves whichever family's fingerprint sensor actually required a
# bigger slice to stay separable at deep stealth.
import sys, json

inst = json.load(sys.stdin)
K = inst["channels"]
benign = inst["benign"]
fp_cap = inst["fp_cap"]
n = inst["n_benign"]

p = fp_cap / K
theta = []
for c in range(K):
    col = [row[c] for row in benign]
    srt = sorted(col)
    k = int(p * n)
    if k <= 0:
        theta.append(srt[-1] + 1.0)
    elif k >= n:
        theta.append(srt[0] - 1.0)
    else:
        theta.append(srt[n - 1 - k])

w = [1.0] * K
tau = 1.0

print(json.dumps({"theta": theta, "w": w, "tau": tau}))
