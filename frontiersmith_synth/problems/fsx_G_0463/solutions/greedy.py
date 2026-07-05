# TIER: greedy
# Shallow greedy tree.  Split criterion: minimise the sample-weighted Gini impurity
# of the two children (a standard CART split), grown top-down to a SMALL fixed depth
# (3) with no leaf-size regularisation.  This carves the single biggest risk region
# but is too shallow to isolate the population's several disjoint risk regions, so it
# beats the majority rule yet leaves accuracy on the table for a deeper builder.
import sys, json

inst = json.load(sys.stdin)
X = inst["X_train"]; y = inst["y_train"]; F = inst["n_features"]
DEPTH_CAP = 3


def gini(c0, c1):
    tot = c0 + c1
    if tot == 0:
        return 0.0
    p = c0 / tot
    return 1.0 - p * p - (1.0 - p) * (1.0 - p)


nodes = []


def build(idx, depth):
    c1 = sum(1 for i in idx if y[i] == 1); c0 = len(idx) - c1
    maj = 0 if c0 >= c1 else 1
    if depth >= DEPTH_CAP or c0 == 0 or c1 == 0 or len(idx) < 2:
        nodes.append({"leaf": maj}); return len(nodes) - 1
    best = None
    for f in range(F):
        vals = sorted(set(X[i][f] for i in idx))
        for k in range(len(vals) - 1):
            t = (vals[k] + vals[k + 1]) / 2.0
            l0 = l1 = r0 = r1 = 0
            for i in idx:
                if X[i][f] <= t:
                    if y[i] == 0: l0 += 1
                    else: l1 += 1
                else:
                    if y[i] == 0: r0 += 1
                    else: r1 += 1
            nl = l0 + l1; nr = r0 + r1
            if nl == 0 or nr == 0:
                continue
            imp = (nl * gini(l0, l1) + nr * gini(r0, r1)) / len(idx)
            if best is None or imp < best[0]:
                best = (imp, f, t)
    if best is None:
        nodes.append({"leaf": maj}); return len(nodes) - 1
    _, bf, bt = best
    L = [i for i in idx if X[i][bf] <= bt]
    R = [i for i in idx if X[i][bf] > bt]
    my = len(nodes); nodes.append(None)
    li = build(L, depth + 1); ri = build(R, depth + 1)
    nodes[my] = {"feature": bf, "threshold": bt, "left": li, "right": ri}
    return my


build(list(range(len(X))), 0)
print(json.dumps({"nodes": nodes}))
