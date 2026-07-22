#!/usr/bin/env python3
"""
Checker for fsx_S_0714 (Grid Librarian).

Input format (<in>):
  line 1:            N M K
  next N lines:       id k a_1 ... a_k w        (errand id; k reads; write addr)
  next M lines:       u v                        (errand u must precede errand v)

Participant output (<out>): N whitespace-separated integers -- a permutation
of the N errand ids, the chosen execution order.

Scoring (minimization): simulate an LRU cart of capacity K over the address
trace induced by the order (each errand's reads, in the given order, then its
write).  F = miss count of the submission.  B = miss count of an internal,
cache-blind topological order (ascending-id Kahn's algorithm), which the
checker builds itself.  Ratio = min(1, B/F) rescaled into [0, 1] the usual way.
"""
import sys
import math
import heapq
from collections import OrderedDict


def fail(reason):
    print("INVALID:", reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_input(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)

    def nx():
        return next(it)

    N = int(nx())
    M = int(nx())
    K = int(nx())
    op_reads = {}
    op_write = {}
    op_ids = []
    for _ in range(N):
        oid = int(nx())
        k = int(nx())
        reads = [int(nx()) for _ in range(k)]
        w = int(nx())
        op_reads[oid] = reads
        op_write[oid] = w
        op_ids.append(oid)
    edges = []
    adj = {o: [] for o in op_ids}
    for _ in range(M):
        u = int(nx())
        v = int(nx())
        edges.append((u, v))
        adj[u].append(v)
    return N, M, K, op_ids, op_reads, op_write, edges, adj


def simulate(order, op_reads, op_write, K):
    od = OrderedDict()
    misses = 0
    for oid in order:
        for a in op_reads[oid]:
            if a in od:
                od.move_to_end(a)
            else:
                misses += 1
                od[a] = True
                if len(od) > K:
                    od.popitem(last=False)
        w = op_write[oid]
        if w in od:
            od.move_to_end(w)
        else:
            misses += 1
            od[w] = True
            if len(od) > K:
                od.popitem(last=False)
    return misses


def kahn_baseline(op_ids, edges):
    indeg = {o: 0 for o in op_ids}
    adj = {o: [] for o in op_ids}
    for u, v in edges:
        indeg[v] += 1
        adj[u].append(v)
    ready = [o for o in op_ids if indeg[o] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        o = heapq.heappop(ready)
        order.append(o)
        for w in adj[o]:
            indeg[w] -= 1
            if indeg[w] == 0:
                heapq.heappush(ready, w)
    return order


def main():
    if len(sys.argv) < 3:
        fail("usage: counter.py <in> <out> <ans>")
    inpath, outpath = sys.argv[1], sys.argv[2]

    N, M, K, op_ids, op_reads, op_write, edges, adj = read_input(inpath)
    op_id_set = set(op_ids)

    try:
        with open(outpath) as f:
            raw = f.read()
    except Exception as e:
        fail(f"cannot read output: {e}")

    toks = raw.split()
    if len(toks) != N:
        fail(f"expected {N} tokens, got {len(toks)}")

    order = []
    seen = set()
    for t in toks:
        try:
            v = int(t)
        except ValueError:
            fail(f"non-integer token {t!r}")
        if not math.isfinite(v):
            fail("non-finite token")
        if v not in op_id_set:
            fail(f"unknown errand id {v}")
        if v in seen:
            fail(f"duplicate errand id {v}")
        seen.add(v)
        order.append(v)

    if seen != op_id_set:
        fail("output is not a permutation of the errand id set")

    pos = {o: i for i, o in enumerate(order)}
    for u, v in edges:
        if pos[u] >= pos[v]:
            fail(f"dependence violated: errand {u} must precede errand {v}")

    F = simulate(order, op_reads, op_write, K)
    base_order = kahn_baseline(op_ids, edges)
    B = simulate(base_order, op_reads, op_write, K)

    F = max(F, 1)
    B = max(B, 1)
    sc = min(1000.0, 100.0 * B / F)
    print("misses=%d baseline_misses=%d" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
