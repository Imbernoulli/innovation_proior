# TIER: trivial
# Reproduces the checker's own internal baseline: meet every typology floor with its
# cheapest alerts, then fill remaining budget scanning alert ids in ascending order.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    test_id = int(next(it)); N = int(next(it)); K = int(next(it)); C = int(next(it))
    alerts = {}
    for _ in range(N):
        aid = int(next(it)); t = int(next(it)); cost = int(next(it)); score = int(next(it))
        alerts[aid] = (t, cost, score)
    min_cover = {}
    for _ in range(K):
        t = int(next(it)); mc = int(next(it))
        min_cover[t] = mc

    by_t = {}
    for aid, (t, cost, score) in alerts.items():
        by_t.setdefault(t, []).append(aid)
    for t in by_t:
        by_t[t].sort(key=lambda a: (alerts[a][1], a))

    sel = set()
    cost_used = 0
    for t in range(1, K + 1):
        need = min_cover.get(t, 0)
        for aid in by_t.get(t, [])[:need]:
            sel.add(aid); cost_used += alerts[aid][1]

    for aid in sorted(alerts):
        if aid in sel:
            continue
        c = alerts[aid][1]
        if cost_used + c <= C:
            sel.add(aid); cost_used += c

    out = [str(len(sel))]
    out.extend(str(a) for a in sorted(sel))
    print("\n".join(out))


if __name__ == "__main__":
    main()
