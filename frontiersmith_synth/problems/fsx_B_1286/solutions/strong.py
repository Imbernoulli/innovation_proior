# TIER: strong
"""The insight: mapping is not "cost with no reward" -- it is a budget item
whose payoff is the WHOLE unlocked subtree, not just its own small
self-mitigation. Framed that way, the problem is a dependency-knapsack
(tree knapsack): each node offers do-nothing / audit / map / map+audit, and
mapping "opens" a children sub-knapsack. Solve it bottom-up with an exact
DP over (node, budget-spent-in-this-subtree), merging sibling subtrees
pairwise, then reconstruct the chosen action sequence top-down. This lets
the solver correctly trade a few direct tier-1 audits for unlocking many
cheap tier-3 audits when that is where the discounted risk mass actually
sits -- exactly the visibility-investment trade-off tier-1-only auditing
can never see."""
import sys


def main():
    sys.setrecursionlimit(10000)
    toks = sys.stdin.read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    idx += 1  # t1n
    budget = int(toks[idx]); idx += 1
    prop = float(toks[idx]); idx += 1
    mapmit = float(toks[idx]); idx += 1
    auditmit = float(toks[idx]); idx += 1

    tier, parent, risk, mapcost, auditcost = {}, {}, {}, {}, {}
    children = {}
    t1ids = []
    for _ in range(n):
        nid = int(toks[idx]); idx += 1
        t = int(toks[idx]); idx += 1
        p = int(toks[idx]); idx += 1
        r = int(toks[idx]); idx += 1
        mc = int(toks[idx]); idx += 1
        ac = int(toks[idx]); idx += 1
        tier[nid] = t; parent[nid] = p; risk[nid] = r
        mapcost[nid] = mc; auditcost[nid] = ac
        children.setdefault(nid, [])
        if p:
            children.setdefault(p, []).append(nid)
        if t == 1:
            t1ids.append(nid)

    B = budget

    def factor(t):
        return prop ** (t - 1)

    def val_audit(v):
        return auditmit * risk[v] * factor(tier[v])

    def val_map(v):
        return mapmit * risk[v] * factor(tier[v])

    def variants_for(v):
        vs = [(0, 0.0, False, "none"), (auditcost[v], val_audit(v), False, "audit")]
        if tier[v] in (1, 2):
            vs.append((mapcost[v], val_map(v), True, "map"))
            vs.append((mapcost[v] + auditcost[v], val_audit(v), True, "mapaudit"))
        return vs

    def merge(prev, kdp):
        new = [0.0] * (B + 1)
        for b in range(B + 1):
            best = 0.0
            pb = prev[b]
            if pb > best:
                best = pb
            for c in range(1, b + 1):
                val = prev[b - c] + kdp[c]
                if val > best:
                    best = val
            new[b] = best
        return new

    def cum_list_for(kids):
        cum = [[0.0] * (B + 1)]
        for kid in kids:
            cum.append(merge(cum[-1], dp[kid]))
        return cum

    dp = {}

    def build_dp(v):
        for c in children.get(v, []):
            build_dp(c)
        combo = cum_list_for(children.get(v, []))[-1]
        arr = [0.0] * (B + 1)
        for b in range(B + 1):
            best = 0.0
            for (c, val, opens, _lab) in variants_for(v):
                if c > b:
                    continue
                tot = val + (combo[b - c] if opens else 0.0)
                if tot > best:
                    best = tot
            arr[b] = best
        dp[v] = arr

    for t in t1ids:
        build_dp(t)

    def decompose(kids, target):
        cum = cum_list_for(kids)
        b = target
        alloc = []
        for i in range(len(kids), 0, -1):
            kid = kids[i - 1]
            prev = cum[i - 1]
            kdp = dp[kid]
            best_c, best_val = 0, prev[b]
            for c in range(0, b + 1):
                val = prev[b - c] + kdp[c]
                if val > best_val:
                    best_val, best_c = val, c
            alloc.append((kid, best_c))
            b -= best_c
        alloc.reverse()
        return alloc

    def reconstruct(v, b):
        kids = children.get(v, [])
        combo = cum_list_for(kids)[-1]
        best_val, best = -1.0, None
        for (c, val, opens, lab) in variants_for(v):
            if c > b:
                continue
            tot = val + (combo[b - c] if opens else 0.0)
            if tot > best_val:
                best_val, best = tot, (c, opens, lab)
        c, opens, lab = best
        out = []
        if lab in ("map", "mapaudit"):
            out.append(f"M {v}")
        if lab in ("audit", "mapaudit"):
            out.append(f"A {v}")
        if opens and kids:
            rem = b - c
            for kid, kb in decompose(kids, rem):
                out.extend(reconstruct(kid, kb))
        return out

    actions = []
    for kid, kb in decompose(t1ids, B):
        actions.extend(reconstruct(kid, kb))

    sys.stdout.write("\n".join(actions) + ("\n" if actions else ""))


if __name__ == "__main__":
    main()
