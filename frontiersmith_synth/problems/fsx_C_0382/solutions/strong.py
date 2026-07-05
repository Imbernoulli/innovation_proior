# TIER: strong
# Order-3 (trigram) model matching the true local calibration law, with
# stupid-backoff (trigram -> bigram -> unigram -> global mode) for contexts
# unseen in the limited training data.  Because the law is local it generalises
# to the longer OOD arrays; the ceiling comes from context coverage, so smarter
# smoothing/backoff can still improve the score (genuinely open-ended).
import sys, json
from collections import Counter

inst = json.load(sys.stdin)
bd = inst["boundary"]
T = {}
B = {}
U = {}
glob = Counter()
for tr in inst["train"]:
    x, y = tr["x"], tr["y"]
    for i in range(len(x)):
        a = x[i - 2] if i > 1 else bd
        b = x[i - 1] if i > 0 else bd
        T.setdefault((a, b, x[i]), Counter())[y[i]] += 1
        B.setdefault((b, x[i]), Counter())[y[i]] += 1
        U.setdefault(x[i], Counter())[y[i]] += 1
        glob[y[i]] += 1
gm = glob.most_common(1)[0][0] if glob else 0

def predict(x):
    out = []
    for i in range(len(x)):
        a = x[i - 2] if i > 1 else bd
        b = x[i - 1] if i > 0 else bd
        if (a, b, x[i]) in T:
            out.append(T[(a, b, x[i])].most_common(1)[0][0])
        elif (b, x[i]) in B:
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
