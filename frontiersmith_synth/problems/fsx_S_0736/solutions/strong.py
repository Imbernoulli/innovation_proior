# TIER: strong
"""
The insight: cost is a property of the SELECTED SET, not of any one stall's choice in
isolation, so per-stall reasoning (the "greedy" tree-DP) is structurally blind to sharing.

Step 1 -- run the same per-stall tree-DP as `greedy` to get a starting selection and, for
free, each stall's stand-alone tree-cost best[i] (needed below).

Step 2 -- CLUSTER stalls by "what would they jointly need if they switched": group every
NON-selected alternate candidate across ALL stalls by its exact children-set signature. Any
signature shared by >= 2 different stalls' alternates identifies a potential coalition: if
those stalls jointly adopted that shared recipe, the referenced ingredient-stalls would be
paid for ONCE instead of once per private alternative. For each such cluster, compare the
coalition's TRUE joint cost (sum of new own-costs + the shared subtree's cost, paid once)
against the sum of the members' current tree-DP costs -- and adopt the switch only if the
whole coalition is strictly cheaper together. A single member switching alone is never
locally attractive (that's exactly the trap) -- only evaluating the GROUP exposes the win.

Step 3 -- safety net: because this is a heuristic exchange move (not a certified optimum),
independently verify the TRUE total cost (via the same reachable-set accounting the checker
uses) of both the tree-DP selection and the switched selection, and output whichever is
actually cheaper. This guarantees strong is never worse than greedy.
"""
import sys
from collections import defaultdict


def main():
    data = sys.stdin.buffer.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]; pos += 1
        return v

    N = int(nxt()); R = int(nxt())
    roots = [int(nxt()) for _ in range(R)]
    classes = []
    for _i in range(N):
        M = int(nxt())
        cands = []
        for _k in range(M):
            cost = int(nxt())
            L = int(nxt())
            children = [int(nxt()) for _ in range(L)]
            cands.append((cost, children))
        classes.append(cands)

    # ---- Step 1: per-stall tree-DP (identical to greedy) ----
    best = [0] * N
    sel_greedy = [0] * N
    for i in range(N - 1, -1, -1):
        b = None
        bk = 0
        for k, (cost, children) in enumerate(classes[i]):
            c = cost
            for ch in children:
                c += best[ch]
            if b is None or c < b:
                b = c; bk = k
        best[i] = b
        sel_greedy[i] = bk

    # ---- Step 2: cluster alternates by shared children-signature ----
    groups = defaultdict(list)  # signature(tuple of children) -> [(i, k), ...]
    for i in range(N):
        cur = sel_greedy[i]
        for k, (cost, children) in enumerate(classes[i]):
            if k == cur or not children:
                continue
            sig = tuple(sorted(children))
            groups[sig].append((i, k))

    sel_strong = list(sel_greedy)
    for sig, members in groups.items():
        if len(members) < 2:
            continue
        new_total = sum(classes[i][k][0] for (i, k) in members) + sum(best[c] for c in sig)
        old_total = sum(best[i] for (i, _k) in members)
        if new_total < old_total:
            for (i, k) in members:
                sel_strong[i] = k

    # ---- Step 3: honest reachable-set cost check (mirrors the checker), keep the cheaper ----
    def total_cost(sel):
        color = [0] * N  # 0 white, 1 gray, 2 black
        for r in roots:
            if color[r] != 0:
                continue
            color[r] = 1
            stack = [(r, 0, classes[r][sel[r]][1])]
            while stack:
                node, idx2, children = stack[-1]
                if idx2 < len(children):
                    stack[-1] = (node, idx2 + 1, children)
                    c = children[idx2]
                    if color[c] == 0:
                        color[c] = 1
                        stack.append((c, 0, classes[c][sel[c]][1]))
                else:
                    color[node] = 2
                    stack.pop()
        return sum(classes[i][sel[i]][0] for i in range(N) if color[i] == 2)

    cost_greedy = total_cost(sel_greedy)
    cost_strong = total_cost(sel_strong)
    final = sel_strong if cost_strong < cost_greedy else sel_greedy

    sys.stdout.write("\n".join(str(x) for x in final) + "\n")


if __name__ == "__main__":
    main()
