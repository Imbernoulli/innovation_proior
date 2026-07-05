# TIER: trivial
import sys, json
from collections import Counter

inst = json.load(sys.stdin)
g = Counter()
for tr in inst["train"]:
    for v in tr["y"]:
        g[v] += 1
m = g.most_common(1)[0][0] if g else 0
q = inst["queries"]
ans = {"predictions": {
    "id": [[m] * len(x) for x in q["id"]],
    "ood": [[m] * len(x) for x in q["ood"]],
}}
print(json.dumps(ans))
