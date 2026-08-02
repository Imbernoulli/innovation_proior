# TIER: strong
"""
The insight: a write's timestamp tells you nothing about whether it is
causally current. What every replica CAN agree on, independent of delivery
order, is each key's causal frontier -- the writes to that key that no
other write to that key happened-after. Recomputing it from vector-clock
dominance (mechanism 1: causal-ordering) turns "who wins this key" into an
exact, order-independent fact rather than a wall-clock accident.

Given the frontiers, the remaining decision (mechanism 2:
conflict-resolution-rule under budget) is: for every multi-writer frontier
that its key allows merging, is spending budget to merge it (earning the
WHOLE frontier's weight) worth more than just taking the frontier's own
best-weighted member for free? That trade-off, across all keys sharing one
budget, is a 0/1 knapsack -- solved exactly with a small DP (instance sizes
are tiny), not "greedy + more iterations".
"""
import sys


def leq(a, b):
    return all(x <= y for x, y in zip(a, b))


def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    R = int(next(it)); K = int(next(it)); N = int(next(it)); BUDGET = int(next(it))
    mtypes = [int(next(it)) for _ in range(K)]
    mcosts = [int(next(it)) for _ in range(K)]
    ops = []
    for _ in range(N):
        replica = int(next(it)); key = int(next(it)); value = int(next(it))
        weight = int(next(it)); _ts = int(next(it))
        vc = [int(next(it)) for _ in range(R)]
        ops.append((replica, key, value, weight, vc))

    by_key = [[] for _ in range(K)]
    for idx, op in enumerate(ops, start=1):
        by_key[op[1]].append(idx)

    frontiers = []
    for k in range(K):
        ids = by_key[k]
        frontier = []
        for i in ids:
            vci = ops[i - 1][4]
            dominated = False
            for j in ids:
                if i == j:
                    continue
                vcj = ops[j - 1][4]
                if leq(vci, vcj) and vci != vcj:
                    dominated = True
                    break
            if not dominated:
                frontier.append(i)
        frontiers.append(frontier)

    # default resolution: pick per key (frontier's own best-weighted member)
    decision = {}  # key -> ("P", ref, value) or ("M", -1, value)
    candidates = []  # (key, cost, gain) for keys eligible for a merge upgrade
    for k in range(K):
        frontier = frontiers[k]
        best_ref = max(frontier, key=lambda i: ops[i - 1][3])
        best_w = ops[best_ref - 1][3]
        decision[k] = ("P", best_ref, ops[best_ref - 1][2])
        if mtypes[k] != 0 and len(frontier) >= 2:
            fr_vals = [ops[i - 1][2] for i in frontier]
            fr_weights = [ops[i - 1][3] for i in frontier]
            merged_val = sum(fr_vals) if mtypes[k] == 1 else max(fr_vals)
            gain = sum(fr_weights) - best_w
            if gain > 0:
                candidates.append((k, mcosts[k], gain, merged_val))

    # 0/1 knapsack DP over candidate merge upgrades, capacity = BUDGET
    if candidates and BUDGET > 0:
        n = len(candidates)
        dp = [[0] * (BUDGET + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            _k, cost, gain, _mv = candidates[i - 1]
            for b in range(BUDGET + 1):
                dp[i][b] = dp[i - 1][b]
                if cost <= b and dp[i - 1][b - cost] + gain > dp[i][b]:
                    dp[i][b] = dp[i - 1][b - cost] + gain
        # backtrack to find which candidates are selected
        b = BUDGET
        for i in range(n, 0, -1):
            _k, cost, gain, _mv = candidates[i - 1]
            if cost <= b and dp[i - 1][b - cost] + gain == dp[i][b] and gain > 0 \
                    and dp[i][b] != dp[i - 1][b]:
                key_sel, _cost, _gain, merged_val = candidates[i - 1]
                decision[key_sel] = ("M", -1, merged_val)
                b -= cost

    out_lines = []
    for k in range(K):
        mode, ref, value = decision[k]
        out_lines.append(f"{k} {mode} {ref} {value}")
    print("\n".join(out_lines))


if __name__ == "__main__":
    main()
