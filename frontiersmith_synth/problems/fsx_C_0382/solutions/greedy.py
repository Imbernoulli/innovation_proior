# TIER: greedy
# Order-2 (bigram) model: correct state depends on the previous neighbour and
# the current one.  Length-invariant, so it transfers to OOD, but it ignores
# the two-neighbour interaction of the true order-3 law -> only partial.
import sys, json
from collections import Counter

inst = json.load(sys.stdin)
bd = inst["boundary"]
B = {}
U = {}
glob = Counter()
for tr in inst["train"]:
    x, y = tr["x"], tr["y"]
    for i in range(len(x)):
        b = x[i - 1] if i > 0 else bd
        B.setdefault((b, x[i]), Counter())[y[i]] += 1
        U.setdefault(x[i], Counter())[y[i]] += 1
        glob[y[i]] += 1
gm = glob.most_common(1)[0][0] if glob else 0

def predict(x):
    out = []
    for i in range(len(x)):
        b = x[i - 1] if i > 0 else bd
        if (b, x[i]) in B:
            out.append(B[(b, x[i])].most_common(1)[0][0])
        elif x[i] in U:
            out.append(U[x[i]].most_common(1)[0][0])
        else:
            out.append(gm)
    return out

q = inst["queries"]
ans = {"predictions": {
    "id": [predict(x) for x in q["id"]],
    "ood": [predict(x) for x in q["ood"]],
}}
print(json.dumps(ans))
