# TIER: greedy
"""The obvious first idea: chase the biggest dollar tickets. Sort ALL transactions by
reported value, descending, and audit them one by one until the budget runs out.

This maximizes the raw dollars sitting in the audited sample -- but it concentrates on
whichever few transactions are largest inside each stratum it touches (a judgmental,
top-of-the-range pick), so it usually fails the representativeness/precision checks
needed to defensibly PROJECT that finding onto the rest of the population."""
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

    rows.sort(key=lambda r: -r[2])
    chosen = []
    spent = 0
    for tid, h, v, cost in rows:
        if spent + cost <= Cmax:
            chosen.append(tid)
            spent += cost

    print(" ".join(map(str, chosen)))


if __name__ == "__main__":
    main()
