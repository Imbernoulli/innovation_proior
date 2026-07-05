# TIER: greedy
# Independent per-station newsvendor critical fractile, then scale the safety part to fit budget.
# Ignores the pooling (tree) structure and the service floor entirely.
import sys, json
from statistics import NormalDist
N = NormalDist(0, 1)
inst = json.load(sys.stdin)
n = inst["n"]; mean = inst["mean"]; std = inst["std"]; h = inst["h"]; p = inst["p"]; B = inst["budget"]
z = [N.inv_cdf(p[i] / (p[i] + h[i])) for i in range(n)]
q = [mean[i] + z[i] * std[i] for i in range(n)]
base = sum(mean); tot = sum(q)
if tot > B:
    safe = tot - base; room = B - base
    f = max(0.0, room / safe) if safe > 0 else 0.0
    q = [mean[i] + f * (q[i] - mean[i]) for i in range(n)]
q = [max(0.0, v) for v in q]
print(json.dumps({"stock": q}))
