# TIER: greedy
# The textbook adaptive-experiment recipe: buy assays by marginal INFORMATION
# GAIN per cost.  Information = the identification entropy of the posterior over
# candidate identities (sum over confusion classes of W_C * H(within-class prior
# weights)).  Add, one at a time, the affordable assay with the largest
# entropy-reduction / cost, until nothing helps or the budget is spent.
#
# This is exactly the trap: information gain depends only on the PRIOR WEIGHTS of
# what an assay separates, never on how far apart the theta values are.  It
# lavishes the budget on cheap high-information assays and, among the premium
# assays, breaks the highest-WEIGHT alias -- which is engineered to be the
# LOW-variance one -- leaving the theta-extreme alias (which dominates the
# estimation error) unbroken.
import sys, json, math

inst = json.load(sys.stdin)
K = inst["K"]; prior = inst["prior"]; read = inst["read"]
M = inst["M"]; cost = inst["cost"]; budget = inst["budget"]


def entropy(ws):
    W = sum(ws)
    if W <= 0:
        return 0.0
    h = 0.0
    for w in ws:
        if w > 0:
            p = w / W
            h -= p * math.log2(p)
    return h


def id_entropy(S):
    groups = {}
    for h in range(K):
        key = tuple(read[j][h] for j in S)
        groups.setdefault(key, []).append(h)
    tot = 0.0
    for members in groups.values():
        tot += sum(prior[h] for h in members) * entropy([prior[h] for h in members])
    return tot


S = []
remaining = set(range(M))
spent = 0
Hcur = id_entropy(S)
while True:
    best = None
    best_ratio = 1e-18
    best_H = Hcur
    for j in sorted(remaining):
        if spent + cost[j] > budget:
            continue
        Hn = id_entropy(S + [j])
        gain = Hcur - Hn
        ratio = gain / cost[j]
        if ratio > best_ratio + 1e-15:
            best_ratio = ratio
            best = j
            best_H = Hn
    if best is None:
        break
    S.append(best)
    remaining.discard(best)
    spent += cost[best]
    Hcur = best_H

print(json.dumps({"probes": sorted(S)}))
