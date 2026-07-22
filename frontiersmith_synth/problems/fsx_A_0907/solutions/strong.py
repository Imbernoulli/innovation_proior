# TIER: strong
import sys, json


def main():
    inst = json.load(sys.stdin)
    n = inst["n"]; K = inst["K"]
    a = inst["a"]; c = inst["c"]; x0 = inst["x0"]
    edges = [(int(e[0]), int(e[1]), float(e[2])) for e in inst["edges"]]

    nbrs = [[] for _ in range(n)]
    S = [0.0] * n
    for i, j, w in edges:
        nbrs[i].append((j, w))
        nbrs[j].append((i, w))
        S[i] += w
        S[j] += w

    x = list(x0)
    order = []
    visited = [False] * n

    def expected_decrease(i):
        # Exact single-coordinate-minimization decrease g_i^2 / (4*(a_i+S_i)):
        # this is the real quantity the coupling graph feeds into (bigger
        # S_i / bigger neighbour mismatch => bigger payoff), not just |g_i|.
        num = a[i] * c[i]
        for j, w in nbrs[i]:
            num += w * x[j]
        g = 2.0 * (a[i] + S[i]) * x[i] - 2.0 * num
        return (g * g) / (4.0 * (a[i] + S[i]))

    def apply_update(i):
        num = a[i] * c[i]
        for j, w in nbrs[i]:
            num += w * x[j]
        x[i] = num / (a[i] + S[i])
        order.append(i)
        visited[i] = True

    # --- coupling-graph diagnosis + Gauss-Southwell, unified: every gauge
    # gets exactly one mandatory visit (K > n guarantees the budget for this),
    # but WHICH not-yet-visited gauge goes first is itself chosen by current
    # expected decrease -- reading the coupling graph rather than assuming a
    # fixed traversal order, since which direction actually propagates useful
    # information depends on the (heterogeneous, planted) edge weights, not
    # on any topological shape (star, chain, ...) alone. Once every gauge has
    # been visited once, the remaining SURPLUS budget is spent purely by
    # Gauss-Southwell: repeatedly re-update whichever gauge currently has the
    # largest exact decrease -- isolated singletons never resurface (their
    # decrease is permanently 0 after their first visit), so every extra
    # update is automatically concentrated on the still-unsettled coupled
    # block(s), spending the scarce budget exactly where the gradient is
    # steep right now ---
    for _ in range(n):
        best_i, best_gain = None, -1.0
        for i in range(n):
            if visited[i]:
                continue
            gain = expected_decrease(i)
            if gain > best_gain:
                best_gain = gain
                best_i = i
        apply_update(best_i)

    remaining = K - len(order)
    for _ in range(max(0, remaining)):
        best_i, best_gain = 0, -1.0
        for i in range(n):
            gain = expected_decrease(i)
            if gain > best_gain:
                best_gain = gain
                best_i = i
        apply_update(best_i)

    order = order[:K]
    if len(order) < K:
        order.extend([0] * (K - len(order)))

    print(json.dumps({"order": order}))


main()
