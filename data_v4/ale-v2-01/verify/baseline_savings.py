#!/usr/bin/env python3
"""Savings (Clarke-Wright, parallel) baseline -- the normalization reference
named in the candidate ("ratio to savings baseline"). No local search, no LNS.
Reads instance on stdin, writes a feasible solution on stdout."""
import sys
import math


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); cap = int(next(it))
    dx = int(next(it)); dy = int(next(it))
    depot = (dx, dy)
    X = [None] * (n + 1); Y = [None] * (n + 1); D = [0] * (n + 1)
    for i in range(1, n + 1):
        X[i] = int(next(it)); Y[i] = int(next(it)); D[i] = int(next(it))

    def dist(a, b):
        ax, ay = (depot if a == 0 else (X[a], Y[a]))
        bx, by = (depot if b == 0 else (X[b], Y[b]))
        return math.hypot(ax - bx, ay - by)

    routes = {c: [c] for c in range(1, n + 1)}
    load = {c: D[c] for c in range(1, n + 1)}
    route_of = {c: c for c in range(1, n + 1)}

    savs = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = dist(i, 0) + dist(0, j) - dist(i, j)
            if s > 0:
                savs.append((s, i, j))
    savs.sort(reverse=True)

    for s, i, j in savs:
        ri, rj = route_of[i], route_of[j]
        if ri == rj:
            continue
        if load[ri] + load[rj] > cap:
            continue
        a = routes[ri]; b = routes[rj]
        if a[-1] != i and a[0] != i:
            continue
        if b[-1] != j and b[0] != j:
            continue
        if a[0] == i:
            a = a[::-1]
        if b[-1] == j:
            b = b[::-1]
        merged = a + b
        routes[ri] = merged
        load[ri] += load[rj]
        for c in merged:
            route_of[c] = ri
        del routes[rj]
        del load[rj]

    out = [str(len(routes))]
    for r in routes.values():
        out.append(str(len(r)) + " " + " ".join(map(str, r)))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
