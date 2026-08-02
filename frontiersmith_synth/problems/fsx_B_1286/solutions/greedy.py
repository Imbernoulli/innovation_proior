# TIER: greedy
"""The obvious first algorithm: repeatedly take the single currently-visible
action (map or audit) with the best IMMEDIATE value/cost ratio, using only
that action's own direct payoff (audit's full mitigation, or map's small
self-mitigation) -- never crediting a map action for the audits it might
unlock later. This is a legitimate, budget-aware, cost-efficiency-aware
knapsack greedy; it just has no lookahead, so it treats "invest in mapping
to reach tier-3" as a weak move (small direct payoff) and spends almost the
whole budget auditing whatever is already visible -- mostly tier-1."""
import sys


def main():
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

    def factor(t):
        return prop ** (t - 1)

    def val_audit(v):
        return auditmit * risk[v] * factor(tier[v])

    def val_map(v):
        return mapmit * risk[v] * factor(tier[v])

    visible = set(t1ids)
    mapped, audited = set(), set()
    remaining = budget
    out = []

    while True:
        best = None
        best_ratio = -1.0
        for v in visible:
            if v not in audited:
                c = auditcost[v]
                gain = val_audit(v) - (val_map(v) if v in mapped else 0.0)
                if c <= remaining and c > 0 and gain > 0:
                    r = gain / c
                    if r > best_ratio:
                        best_ratio, best = r, ("A", v, c)
            if tier[v] in (1, 2) and v not in mapped:
                c = mapcost[v]
                gain = val_map(v)
                if c <= remaining and c > 0 and gain > 0:
                    r = gain / c
                    if r > best_ratio:
                        best_ratio, best = r, ("M", v, c)
        if best is None:
            break
        op, v, c = best
        remaining -= c
        out.append(f"{op} {v}")
        if op == "A":
            audited.add(v)
        else:
            mapped.add(v)
            for ch in children.get(v, []):
                visible.add(ch)

    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
