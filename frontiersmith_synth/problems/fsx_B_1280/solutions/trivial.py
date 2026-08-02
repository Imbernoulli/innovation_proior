# TIER: trivial
"""Lazy audit: fully census the SINGLE stratum that is cheapest to audit completely
(minimum total audit cost among the K strata), and touch nothing else. This is exactly
the checker's own internal calibration baseline -- it reproduces ~0.1 by construction."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    nxt = lambda: next(it)

    t = int(nxt()); N = int(nxt()); K = int(nxt()); Cmax = int(nxt())
    thresh = float(nxt())
    for _ in range(K):
        nxt(); nxt()
    rows = []
    for _ in range(N):
        tid = int(nxt()); h = int(nxt()); v = int(nxt()); cost = int(nxt())
        rows.append((tid, h, v, cost))

    members = {h: [r for r in rows if r[1] == h] for h in range(K)}
    full_cost = {h: sum(r[3] for r in members[h]) for h in range(K)}
    h0 = min(full_cost, key=lambda h: full_cost[h])

    chosen = [r[0] for r in members[h0]]
    print(" ".join(map(str, chosen)))


if __name__ == "__main__":
    main()
