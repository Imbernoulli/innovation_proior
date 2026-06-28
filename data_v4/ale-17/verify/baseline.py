#!/usr/bin/env python3
"""Trivial baseline solver for ale-17: greedy nearest-neighbour split.

Reads an instance from stdin, writes K routes. Used purely as the feasibility /
beat-the-baseline reference for self-verification. It is intentionally simple:
sort customers by angle around the depot and fill vehicles in order under
capacity (a plain sweep with NO local search), which is the natural "toy"
construction. Always feasible given sum(dem) <= K*Q and max(dem) <= Q.
"""
import sys
import math


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    n = int(next(it)); K = int(next(it)); Q = int(next(it))
    dx = float(next(it)); dy = float(next(it))
    xs = [dx]; ys = [dy]; dem = [0]
    for _ in range(n):
        xs.append(float(next(it))); ys.append(float(next(it))); dem.append(int(next(it)))

    order = sorted(range(1, n + 1),
                   key=lambda i: math.atan2(ys[i] - dy, xs[i] - dx))
    routes = [[] for _ in range(K)]
    load = [0] * K
    v = 0
    for c in order:
        while v < K and load[v] + dem[c] > Q:
            v += 1
        if v >= K:
            # place in least-loaded route that fits
            best = -1
            for k in range(K):
                if load[k] + dem[c] <= Q and (best < 0 or load[k] < load[best]):
                    best = k
            if best < 0:
                best = min(range(K), key=lambda k: load[k])
            routes[best].append(c); load[best] += dem[c]
        else:
            routes[v].append(c); load[v] += dem[c]

    out = "\n".join(" ".join(map(str, r)) for r in routes)
    sys.stdout.write(out + "\n")


if __name__ == "__main__":
    main()
