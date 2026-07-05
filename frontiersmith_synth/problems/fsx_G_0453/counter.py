import sys

# Format D checker -- SWAP-count for qubit routing onto a coupling map.
#
#   1) Parse the instance from <in>: coupling graph (P physical qubits, E undirected
#      edges) and an ORDERED schedule of M two-qubit interactions on L logical qubits.
#   2) Parse the participant artifact from <out>:
#         P integers  place[phys] = logical on that physical qubit (-1 if empty)
#         then, for each of the M gates in order:
#             k  followed by  k  pairs  (p q)   -- swaps to apply BEFORE that gate
#      A swap (p,q) exchanges the logical contents of physical qubits p,q and is legal
#      only if {p,q} is a coupling edge.  After a gate's swaps are applied, the gate's
#      two logical qubits MUST reside on two physically-adjacent qubits (exact validity
#      gate) -- otherwise the routing is infeasible and scores 0.
#   3) Objective (minimize) = total number of inserted SWAPs.
#   4) Baseline B = SWAP count of the checker's own identity-placement + greedy
#      shortest-path router (a trivial feasible construction).
#      Ratio = min(1, 0.1 * B / F).  Fewer swaps -> higher ratio.
#
# Deterministic exact-integer scoring; nothing is timed.

MAX_TOKENS = 4_000_000


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def bfs_dist(adj, P, src):
    dist = [-1] * P
    dist[src] = 0
    q = [src]
    h = 0
    while h < len(q):
        u = q[h]; h += 1
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def baseline_swaps(P, L, adj, dist, gates):
    """Identity placement + greedy: for each gate, move logical a one step at a time
    along a shortest path toward logical b (smallest-index next hop) until adjacent."""
    loc = list(range(L)) + [-1] * (P - L)   # loc[logical] = physical (index < L valid)
    occ = [-1] * P
    for i in range(L):
        occ[i] = i
    total = 0
    for (a, b) in gates:
        pa = loc[a]; pb = loc[b]
        while dist[pa][pb] > 1:
            nxt = None
            for n in adj[pa]:            # adj lists are pre-sorted ascending
                if dist[n][pb] == dist[pa][pb] - 1:
                    nxt = n
                    break
            la = occ[pa]; ln = occ[nxt]
            occ[pa] = ln; occ[nxt] = la
            if la != -1:
                loc[la] = nxt
            if ln != -1:
                loc[ln] = pa
            total += 1
            pa = loc[a]; pb = loc[b]
    return total


def main():
    inp = open(sys.argv[1]).read().split()
    it = iter(inp)
    try:
        P = int(next(it)); E = int(next(it)); M = int(next(it)); L = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= P <= 400 and 0 <= E <= P * P and 1 <= M <= 100000 and 1 <= L <= P):
        fail("bad dims")

    adj_set = [set() for _ in range(P)]
    try:
        for _ in range(E):
            u = int(next(it)); v = int(next(it))
            if not (0 <= u < P and 0 <= v < P) or u == v:
                fail("bad edge")
            adj_set[u].add(v)
            adj_set[v].add(u)
    except StopIteration:
        fail("truncated edges")
    adj = [sorted(s) for s in adj_set]

    gates = []
    try:
        for _ in range(M):
            a = int(next(it)); b = int(next(it))
            if not (0 <= a < L and 0 <= b < L) or a == b:
                fail("bad gate")
            gates.append((a, b))
    except StopIteration:
        fail("truncated gates")

    # all-pairs shortest-path distances on the coupling graph
    dist = [bfs_dist(adj, P, s) for s in range(P)]
    for s in range(P):
        for t in range(P):
            if dist[s][t] == -1:
                fail("coupling graph disconnected")

    B = baseline_swaps(P, L, adj, dist, gates)
    if B <= 0:
        fail("degenerate: baseline needs no swaps")

    # ---------- parse participant output ----------
    out = open(sys.argv[2]).read().split()
    if not out:
        fail("empty output")
    if len(out) > MAX_TOKENS:
        fail("output too large")

    jt = iter(out)

    def nxt_int(what):
        try:
            tok = next(jt)
        except StopIteration:
            fail("truncated output (%s)" % what)
        try:
            return int(tok)   # rejects nan/inf/floats -> non-finite guard
        except Exception:
            fail("non-integer token (%s): %r" % (what, tok))

    # placement line: P integers, a valid assignment of logicals 0..L-1
    place = [nxt_int("placement") for _ in range(P)]
    occ = [-1] * P
    loc = [-1] * L
    seen = [False] * L
    for phys in range(P):
        val = place[phys]
        if val == -1:
            continue
        if not (0 <= val < L):
            fail("placement out of range at phys %d" % phys)
        if seen[val]:
            fail("logical %d placed twice" % val)
        seen[val] = True
        occ[phys] = val
        loc[val] = phys
    if not all(seen):
        fail("not all logical qubits placed")

    total = 0
    for gi, (a, b) in enumerate(gates):
        k = nxt_int("swapcount g%d" % gi)
        if not (0 <= k <= P * P):
            fail("bad swap count %d at gate %d" % (k, gi))
        for _ in range(k):
            p = nxt_int("swap-p g%d" % gi)
            q = nxt_int("swap-q g%d" % gi)
            if not (0 <= p < P and 0 <= q < P):
                fail("swap endpoint out of range at gate %d" % gi)
            if q not in adj_set[p]:
                fail("illegal swap (%d,%d): not a coupling edge, gate %d" % (p, q, gi))
            lp = occ[p]; lq = occ[q]
            occ[p] = lq; occ[q] = lp
            if lp != -1:
                loc[lp] = q
            if lq != -1:
                loc[lq] = p
            total += 1
        # validity gate: the two logical qubits must now be physically adjacent
        pa = loc[a]; pb = loc[b]
        if pb not in adj_set[pa]:
            fail("gate %d (logical %d,%d) not executable: phys %d,%d not adjacent"
                 % (gi, a, b, pa, pb))

    # reject any trailing tokens -> strict schema
    if next(jt, None) is not None:
        fail("extra trailing tokens")

    F = total
    ratio = min(1.0, 0.1 * B / max(1, F))
    print("B=%d F=%d Ratio: %.6f" % (B, F, ratio))


if __name__ == "__main__":
    main()
