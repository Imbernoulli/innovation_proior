# TIER: trivial
# Single fixed sensor (sensor 0), thresholded at the FULL private 1% false-
# positive allowance; every other sensor is switched off (weight 0). This
# reproduces the evaluator's own weak reference exactly, so it scores ~0.1:
# it only ever catches whichever family happens to fingerprint sensor 0.
import sys, json

inst = json.load(sys.stdin)
K = inst["channels"]
benign = inst["benign"]
fp_cap = inst["fp_cap"]
n = inst["n_benign"]

col0 = [row[0] for row in benign]
srt = sorted(col0)
k = int(fp_cap * n)
if k <= 0:
    theta0 = srt[-1] + 1.0
elif k >= n:
    theta0 = srt[0] - 1.0
else:
    theta0 = srt[n - 1 - k]

theta = [theta0] + [1e9] * (K - 1)
w = [1.0] + [0.0] * (K - 1)
tau = 1.0

print(json.dumps({"theta": theta, "w": w, "tau": tau}))
