#!/usr/bin/env python3
"""Trivial baseline solver: earliest-start serial SGS with the input-order list.

Reads an instance on stdin, writes one start time per task (input order) on
stdout. This is the same construction the scorer normalizes against, so a
correct solver must beat its score (i.e. produce a strictly shorter makespan on
average). Used only for self-verification.
"""
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); R = int(next(it))
    caps = [int(next(it)) for _ in range(R)]
    dur = [0] * n
    demand = [[0] * R for _ in range(n)]
    preds = [[] for _ in range(n)]
    for i in range(n):
        dur[i] = int(next(it))
        for k in range(R):
            demand[i][k] = int(next(it))
        p = int(next(it))
        for _ in range(p):
            preds[i].append(int(next(it)) - 1)
    if n == 0:
        return
    indeg = [len(preds[i]) for i in range(n)]
    succ = [[] for _ in range(n)]
    for i in range(n):
        for j in preds[i]:
            succ[j].append(i)
    starts = [0] * n
    scheduled = [False] * n
    horizon_cap = sum(dur) + 1
    usage = [[0] * (horizon_cap + 1) for _ in range(R)]

    def earliest(i, t0):
        d = dur[i]; t = t0
        while True:
            ok = True
            for k in range(R):
                dem = demand[i][k]
                if dem == 0:
                    continue
                for tt in range(t, t + d):
                    if usage[k][tt] + dem > caps[k]:
                        ok = False; break
                if not ok:
                    break
            if ok:
                return t
            t += 1

    for _ in range(n):
        cand = -1
        for i in range(n):
            if not scheduled[i] and indeg[i] == 0:
                cand = i; break
        t0 = 0
        for j in preds[cand]:
            t0 = max(t0, starts[j] + dur[j])
        s = earliest(cand, t0)
        starts[cand] = s
        scheduled[cand] = True
        for k in range(R):
            dem = demand[cand][k]
            if dem:
                for tt in range(s, s + dur[cand]):
                    usage[k][tt] += dem
        for c in succ[cand]:
            indeg[c] -= 1

    sys.stdout.write(" ".join(str(x) for x in starts) + "\n")


if __name__ == "__main__":
    main()
