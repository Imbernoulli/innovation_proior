# TIER: strong
# Insight: value an assay by the theta-VARIANCE it removes from the surviving
# confusion classes -- the quantity actually scored -- not by its marginal
# information under the current posterior.  So optimise the true objective
#     J(S) = within-class weighted variance of theta  +  gamma * cost(S)
# directly.  Greedily add the assay with the best objective improvement, then run
# a budget-respecting local search (single remove / add / swap) to escape the
# greedy local optimum.  This spends the budget to break the theta-EXTREME
# aliases that dominate the error (and declines cheap information that barely
# moves theta), so it beats information-per-cost and still leaves headroom for a
# stronger search above it.
import sys, json

inst = json.load(sys.stdin)
K = inst["K"]; prior = inst["prior"]; theta = inst["theta"]; read = inst["read"]
M = inst["M"]; cost = inst["cost"]; budget = inst["budget"]; gamma = inst["gamma"]


def residual(S):
    groups = {}
    for h in range(K):
        key = tuple(read[j][h] for j in S)
        groups.setdefault(key, []).append(h)
    total = 0.0
    for members in groups.values():
        W = sum(prior[h] for h in members)
        if W <= 0:
            continue
        mean = sum(prior[h] * theta[h] for h in members) / W
        for h in members:
            total += prior[h] * (theta[h] - mean) ** 2
    return total


def cost_of(S):
    return sum(cost[j] for j in S)


def J(S):
    return residual(sorted(S)) + gamma * cost_of(S)


S = set()
# greedy construction by best objective improvement
while True:
    cur = J(S)
    best = None
    best_J = cur
    for j in range(M):
        if j in S:
            continue
        if cost_of(S) + cost[j] > budget:
            continue
        nj = J(S | {j})
        if nj < best_J - 1e-12:
            best_J = nj
            best = j
    if best is None:
        break
    S.add(best)

# local search: single remove / add / swap moves
improved = True
while improved:
    improved = False
    cur = J(S)
    for j in list(S):
        nj = J(S - {j})
        if nj < cur - 1e-12:
            S.discard(j)
            improved = True
            cur = nj
    for j in range(M):
        if j in S:
            continue
        if cost_of(S) + cost[j] > budget:
            continue
        nj = J(S | {j})
        if nj < cur - 1e-12:
            S.add(j)
            improved = True
            cur = nj
    for jo in list(S):
        for ji in range(M):
            if ji in S:
                continue
            Sn = (S - {jo}) | {ji}
            if cost_of(Sn) > budget:
                continue
            nj = J(Sn)
            if nj < cur - 1e-12:
                S = Sn
                improved = True
                cur = nj
                break

print(json.dumps({"probes": sorted(S)}))
