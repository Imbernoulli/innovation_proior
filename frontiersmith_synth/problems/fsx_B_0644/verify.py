#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- exact k-terminal reliability checker.

Reads the network + budget from <in>, an upgrade allocation from <out>,
validates feasibility strictly, then computes the EXACT probability that all
k terminals lie in one connected component (brute force over all 2^m
edge-alive/dead states -- m is kept small by the generator), normalized
against the checker's own zero-upgrade baseline construction.
"""
import sys
import math


def read_instance(path):
    toks = open(path).read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    m = int(toks[idx]); idx += 1
    k = int(toks[idx]); idx += 1
    B = int(toks[idx]); idx += 1
    terminals = [int(toks[idx + i]) for i in range(k)]; idx += k
    edges = []
    for _ in range(m):
        u, v, p0, c, mu = (int(toks[idx + j]) for j in range(5))
        idx += 5
        edges.append((u, v, p0, c, mu))
    return n, m, k, B, terminals, edges


class DSU:
    __slots__ = ("p",)

    def __init__(self, n):
        self.p = list(range(n + 1))

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def reliability(n, edges, alive_prob, terminals):
    m = len(edges)
    total = 0.0
    t0 = terminals[0]
    rest = terminals[1:]
    for mask in range(1 << m):
        w = 1.0
        dsu = DSU(n)
        for i in range(m):
            pa = alive_prob[i]
            if (mask >> i) & 1:
                w *= pa
                if w == 0.0:
                    break
                u, v, _, _, _ = edges[i]
                dsu.union(u, v)
            else:
                w *= (1.0 - pa)
                if w == 0.0:
                    break
        if w == 0.0:
            continue
        root0 = dsu.find(t0)
        if all(dsu.find(t) == root0 for t in rest):
            total += w
    return total


def bad(msg):
    print("Ratio: 0.0  # " + msg)
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    n, m, k, B, terminals, edges = read_instance(inf)

    try:
        raw = open(outf).read()
    except Exception:
        bad("cannot read output")
        return
    toks = raw.split()
    if len(toks) != m:
        bad(f"expected {m} integers, got {len(toks)}")
        return
    us = []
    for t in toks:
        try:
            v = int(t)
        except ValueError:
            bad("non-integer token")
            return
        us.append(v)

    for i, (u, v, p0, c, mu) in enumerate(edges):
        if us[i] < 0 or us[i] > mu:
            bad(f"edge {i} upgrade level {us[i]} out of range [0,{mu}]")
            return

    total_cost = sum(edges[i][3] * us[i] for i in range(m))
    if total_cost > B:
        bad(f"total cost {total_cost} exceeds budget {B}")
        return

    alive_sol = []
    alive_base = []
    for i, (u, v, p0, c, mu) in enumerate(edges):
        p0f = p0 / 1000.0
        alive_sol.append(1.0 - p0f * (0.5 ** us[i]))
        alive_base.append(1.0 - p0f)

    F = reliability(n, edges, alive_sol, terminals)
    Bref = reliability(n, edges, alive_base, terminals)

    if not (math.isfinite(F) and math.isfinite(Bref)):
        bad("non-finite reliability")
        return

    sc = min(1000.0, 100.0 * F / max(1e-9, Bref))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
