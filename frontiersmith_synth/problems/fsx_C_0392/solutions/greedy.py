# TIER: greedy
# Pure marginal-association thresholding.  For every unordered pair of variables
# compute the empirical mutual information; if it exceeds a fixed threshold, draw
# a link and orient it by index order (i -> j for i < j).  This recovers much of
# the skeleton but (a) keeps SPURIOUS transitive links (A and C look associated
# whenever A -> B -> C) and (b) orients by an index order that carries NO causal
# information (labels are randomly permuted), so roughly half the links are
# REVERSED.  It beats predicting nothing on sparse regions but is far from strong.
import sys, json, math

inst = json.load(sys.stdin)
n = inst["n"]
data = inst["data"]
m = len(data)

cols = [[row[i] for row in data] for i in range(n)]


def mutual_info(i, j):
    ci = cols[i]
    cj = cols[j]
    joint = {}
    mi_c = {}
    mj_c = {}
    for a, b in zip(ci, cj):
        joint[(a, b)] = joint.get((a, b), 0) + 1
        mi_c[a] = mi_c.get(a, 0) + 1
        mj_c[b] = mj_c.get(b, 0) + 1
    mi = 0.0
    for (a, b), c in joint.items():
        pab = c / m
        pa = mi_c[a] / m
        pb = mj_c[b] / m
        mi += pab * math.log(pab / (pa * pb))
    return mi


THR = 0.02
edges = []
for i in range(n):
    for j in range(i + 1, n):
        if mutual_info(i, j) > THR:
            edges.append([i, j])          # orient by index order (uninformative)

print(json.dumps({"edges": edges}))
