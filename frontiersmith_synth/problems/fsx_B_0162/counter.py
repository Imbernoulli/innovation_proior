#!/usr/bin/env python3
"""counter.py <in> <out> <ans>  -- deterministic op-counter for the QAOA SWAP-routing problem.

Verifies EXACT circuit equivalence (every required logical interaction is executed
on physically adjacent qubits, exactly once, under a correctly-tracked permutation
of the initial identity placement), then counts inserted SWAP gates.  Fewer SWAPs
is better.  Prints `Ratio: r` with r in [0,1]; any feasibility violation -> 0.0.
"""
import sys
from collections import deque


def bfs_dist(adj, N, s):
    d = [-1] * N
    d[s] = 0
    q = deque([s])
    while q:
        x = q.popleft()
        for y in adj[x]:
            if d[y] < 0:
                d[y] = d[x] + 1
                q.append(y)
    return d


def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); K = int(next(it))
    edges = set()
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u = int(next(it)); v = int(next(it))
        e = (min(u, v), max(u, v))
        edges.add(e)
        adj[u].append(v); adj[v].append(u)
    req = []
    for _ in range(K):
        a = int(next(it)); b = int(next(it))
        req.append((min(a, b), max(a, b)))
    return N, edges, adj, req


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    inf, outf = sys.argv[1], sys.argv[2]
    N, edges, adj, req = read_instance(inf)
    req_set = set(req)

    # ---- internal baseline B = route-then-restore each interaction from identity ----
    dist = [bfs_dist(adj, N, s) for s in range(N)]
    B = 0
    for a, b in req:
        d = dist[a][b]
        if d < 0:
            fail("instance disconnected")
        B += 2 * (d - 1)

    # ---- simulate the participant's straight-line program ----
    occ = list(range(N))   # occ[phys] = logical qubit currently on that physical qubit
    loc = list(range(N))   # loc[logical] = physical qubit
    done = set()
    swaps = 0

    try:
        text = open(outf).read()
    except Exception:
        fail("no output")

    for ln in text.split("\n"):
        s = ln.split()
        if not s:
            continue
        tag = s[0]
        if tag == "S":
            if len(s) != 3:
                fail("malformed S line")
            try:
                u = int(s[1]); v = int(s[2])
            except ValueError:
                fail("non-integer SWAP qubit")
            if not (0 <= u < N and 0 <= v < N) or u == v:
                fail("SWAP qubit out of range")
            if (min(u, v), max(u, v)) not in edges:
                fail("SWAP on non-adjacent qubits (not a coupling-map edge)")
            la, lb = occ[u], occ[v]
            occ[u], occ[v] = lb, la
            loc[la], loc[lb] = v, u
            swaps += 1
        elif tag == "G":
            if len(s) != 3:
                fail("malformed G line")
            try:
                a = int(s[1]); b = int(s[2])
            except ValueError:
                fail("non-integer interaction qubit")
            key = (min(a, b), max(a, b))
            if key not in req_set:
                fail("interaction is not in the required cost layer")
            if key in done:
                fail("interaction executed more than once")
            pa, pb = loc[a], loc[b]
            if (min(pa, pb), max(pa, pb)) not in edges:
                fail("interaction qubits are not physically adjacent")
            done.add(key)
        else:
            fail("unknown instruction '%s'" % tag)

    if done != req_set:
        fail("not all required interactions executed (%d/%d)" % (len(done), len(req_set)))

    F = swaps
    if B <= 0:
        # degenerate: every required pair already adjacent -> optimum is 0 SWAPs
        r = 1.0 if F == 0 else 0.0
        print("SWAPs=%d baseline=%d Ratio: %.6f" % (F, B, r))
        return

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("SWAPs=%d baseline=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
