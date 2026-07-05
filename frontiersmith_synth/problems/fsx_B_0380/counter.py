#!/usr/bin/env python3
"""Format D checker -- qubit-routing / SWAP-count minimization (warehouse skin).

  1) Parse the floor graph + ordered handoff tasks from <in>.
  2) Parse the participant's routed program from <out>:
         MAP p_0 p_1 ... p_{V-1}      (permutation: bay of each robot)
         then a sequence of steps, each either
             S u v                    (SWAP robots on ADJACENT bays u,v)
             G                         (execute the NEXT pending handoff)
  3) EQUIVALENCE / FEASIBILITY gate (any violation -> Ratio: 0.0):
         - MAP is a permutation of 0..V-1
         - every S is on an existing edge
         - every G finds its two robots on ADJACENT bays
         - exactly m G's, consumed in the given order, nothing left over
  4) Objective (minimize) = number of S steps (SWAPs) = F.
     Internal baseline B = identity placement + per-task shortest-path routing.
     Ratio = min(1, 0.1 * B / F).
"""
import sys
from collections import deque

MAX_STEPS = 5_000_000


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    tok = open(path).read().split()
    it = iter(tok)
    V = int(next(it)); E = int(next(it)); m = int(next(it))
    adj = [set() for _ in range(V)]
    for _ in range(E):
        u = int(next(it)); v = int(next(it))
        adj[u].add(v); adj[v].add(u)
    tasks = []
    for _ in range(m):
        a = int(next(it)); b = int(next(it))
        tasks.append((a, b))
    return V, adj, tasks


def bfs_path(adj, src, dst):
    """Deterministic shortest path src->dst (sorted-neighbour tie-break)."""
    if src == dst:
        return [src]
    par = {src: src}
    q = deque([src])
    nbr = [sorted(a) for a in adj]
    while q:
        x = q.popleft()
        for y in nbr[x]:
            if y not in par:
                par[y] = x
                if y == dst:
                    path = [dst]
                    while path[-1] != src:
                        path.append(par[path[-1]])
                    path.reverse()
                    return path
                q.append(y)
    return None


def baseline_swaps(V, adj, tasks):
    """Identity placement; for each task move robot `a` along a shortest path
    until adjacent to `b`.  Returns total SWAP count (>=0)."""
    pos = list(range(V))          # pos[robot] = bay
    occ = list(range(V))          # occ[bay]   = robot
    total = 0
    for (a, b) in tasks:
        pa, pb = pos[a], pos[b]
        path = bfs_path(adj, pa, pb)   # bays; connected grid -> always exists
        # move a to path[-2] (adjacent to pb)
        cur = pa
        for k in range(1, len(path) - 1):
            nxt = path[k]
            ra, rn = occ[cur], occ[nxt]
            occ[cur], occ[nxt] = rn, ra
            pos[ra], pos[rn] = nxt, cur
            cur = nxt
            total += 1
    return total


def main():
    V, adj, tasks = read_instance(sys.argv[1])
    m = len(tasks)

    B = baseline_swaps(V, adj, tasks)
    if B <= 0:
        B = 1  # degenerate guard (does not occur for generated instances)

    out = open(sys.argv[2]).read().split()
    if not out:
        fail("empty output")
    if len(out) > MAX_STEPS:
        fail("output too long")

    i = 0
    if out[i] != "MAP":
        fail("missing MAP header")
    i += 1
    if len(out) < 1 + V:
        fail("truncated MAP")
    try:
        placement = [int(out[i + j]) for j in range(V)]
    except Exception:
        fail("non-integer MAP")
    i += V
    if sorted(placement) != list(range(V)):
        fail("MAP is not a permutation of 0..V-1")

    pos = [0] * V          # pos[robot] = bay
    occ = [0] * V          # occ[bay]   = robot
    for robot in range(V):
        bay = placement[robot]
        pos[robot] = bay
        occ[bay] = robot

    swaps = 0
    gate_idx = 0
    n = len(out)
    while i < n:
        tk = out[i]
        if tk == "S":
            if i + 2 >= n:
                fail("truncated SWAP")
            try:
                u = int(out[i + 1]); v = int(out[i + 2])
            except Exception:
                fail("non-integer SWAP args")
            i += 3
            if not (0 <= u < V and 0 <= v < V):
                fail("SWAP bay out of range")
            if v not in adj[u]:
                fail("SWAP on non-adjacent bays %d,%d" % (u, v))
            ru, rv = occ[u], occ[v]
            occ[u], occ[v] = rv, ru
            pos[ru], pos[rv] = v, u
            swaps += 1
        elif tk == "G":
            i += 1
            if gate_idx >= m:
                fail("more G's than handoffs")
            a, b = tasks[gate_idx]
            pa, pb = pos[a], pos[b]
            if pb not in adj[pa]:
                fail("handoff %d (robots %d,%d) not on adjacent bays" % (gate_idx, a, b))
            gate_idx += 1
        else:
            fail("unexpected token '%s'" % tk[:16])

    if gate_idx != m:
        fail("only %d of %d handoffs executed" % (gate_idx, m))

    F = swaps
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("swaps=%d baseline=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
