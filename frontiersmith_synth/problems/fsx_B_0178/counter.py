#!/usr/bin/env python3
"""counter.py <in> <out> <ans>   (ans ignored)

Deterministic op-count scorer for the drone-swarm SWAP-routing (quantum
transpilation) problem.

The participant program (stdout) is a straight-line sequence of ops:
    SWAP p q     swap the drones currently on directly-linked hubs p,q  (cost 1)
    APPLY p q    execute the hand-off between the two drones sitting on
                 directly-linked hubs p,q                                (cost 0)
    END          optional terminator; blank lines ignored
Drone i starts parked on hub i (identity layout).

FEASIBILITY (any violation -> Ratio: 0.0):
  * op tokens well-formed, hub ids in range
  * SWAP/APPLY only on a real mesh edge (p != q, {p,q} in edges)
  * every APPLY targets a still-unexecuted REQUIRED hand-off (as an unordered
    pair of the two drones currently on p,q); no duplicate, no non-required
  * exactly the required set of hand-offs is executed (functional equivalence)

OBJECTIVE (minimize): number of SWAP ops.
BASELINE B (built here): route each required pair independently on a fresh
identity layout via a deterministic shortest path, apply, then undo the swaps
=> B = sum over pairs of 2*(dist-1).  Score (minimization):
    sc = min(1000, 100 * B / max(1e-9, F));  Ratio = sc/1000
"""
import sys
from collections import deque


def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    n = int(next(it)); m = int(next(it)); k = int(next(it))
    adj = [set() for _ in range(n)]
    edges = set()
    for _ in range(m):
        u = int(next(it)); v = int(next(it))
        adj[u].add(v); adj[v].add(u)
        edges.add(frozenset((u, v)))
    required = []
    for _ in range(k):
        a = int(next(it)); b = int(next(it))
        required.append(frozenset((a, b)))
    return n, edges, adj, required


def bfs_path(adj_sorted, s, t, n):
    par = [-2] * n
    par[s] = -1
    dq = deque([s])
    while dq:
        u = dq.popleft()
        if u == t:
            break
        for v in adj_sorted[u]:
            if par[v] == -2:
                par[v] = u
                dq.append(v)
    if par[t] == -2:
        return None
    path = []
    x = t
    while x != -1:
        path.append(x)
        x = par[x]
    path.reverse()
    return path


def baseline_swaps(n, adj, required):
    adj_sorted = [sorted(adj[u]) for u in range(n)]
    total = 0
    for pair in required:
        a, b = tuple(pair)
        # fresh identity layout each pair -> drone a on hub a, drone b on hub b
        path = bfs_path(adj_sorted, a, b, n)
        if path is None or len(path) < 3:
            continue
        s = len(path) - 2  # forward swaps to make them adjacent
        total += 2 * s     # forward + undo
    return total


def fail(reason):
    print("Ratio: 0.0  (%s)" % reason)
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    n, edges, adj, required = read_instance(inf)

    req_set = set(required)
    if len(req_set) != len(required):
        # instance guarantees uniqueness; defensive
        req_set = set(required)

    pos = list(range(n))   # pos[drone] = hub
    occ = list(range(n))   # occ[hub]   = drone
    applied = set()
    swaps = 0

    data = open(outf).read().split("\n")
    for raw in data:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        op = parts[0].upper()
        if op == "END":
            break
        if op not in ("SWAP", "APPLY"):
            fail("unknown op '%s'" % parts[0])
        if len(parts) != 3:
            fail("op '%s' needs 2 args" % op)
        try:
            p = int(parts[1]); q = int(parts[2])
        except ValueError:
            fail("non-integer hub id")
        if not (0 <= p < n and 0 <= q < n):
            fail("hub id out of range")
        if p == q:
            fail("op on identical hubs")
        if frozenset((p, q)) not in edges:
            fail("op on non-linked hubs %d-%d" % (p, q))
        if op == "SWAP":
            da, db = occ[p], occ[q]
            occ[p], occ[q] = db, da
            pos[da], pos[db] = q, p
            swaps += 1
        else:  # APPLY
            da, db = occ[p], occ[q]
            pair = frozenset((da, db))
            if pair not in req_set:
                fail("APPLY on non-required hand-off %d-%d" % (da, db))
            if pair in applied:
                fail("duplicate APPLY of hand-off %d-%d" % (da, db))
            applied.add(pair)

    if applied != req_set:
        fail("executed %d/%d required hand-offs" % (len(applied), len(req_set)))

    F = swaps
    B = baseline_swaps(n, adj, required)
    if B <= 0:
        # no routing was actually required; give full credit for a valid solution
        print("Ratio: 1.0  (baseline 0, F=%d)" % F)
        sys.exit(0)
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("F=%d B=%d  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
