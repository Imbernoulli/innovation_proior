# TIER: trivial
# Predict the single most-frequent training berth code for every UNLOAD.
# Ignores the LIFO stack entirely -> calibrated ~0.1 baseline.
import sys, json
from collections import Counter

inst = json.load(sys.stdin)
g = Counter()
for tr in inst["train"]:
    for v in tr["codes"]:
        g[v] += 1
m = g.most_common(1)[0][0] if g else 0

def n_unloads(moves):
    return sum(1 for x in moves if x == -1)

q = inst["queries"]
ans = {"predictions": {
    "id": [[m] * n_unloads(mv) for mv in q["id"]],
    "ood": [[m] * n_unloads(mv) for mv in q["ood"]],
}}
print(json.dumps(ans))
