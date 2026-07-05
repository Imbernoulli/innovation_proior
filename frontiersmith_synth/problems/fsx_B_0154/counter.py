#!/usr/bin/env python3
"""counter.py <in> <out> <ans>   (Format D: op-count / SWAP-count)

The participant submits a routed program that executes every required logical
interaction (ZZ maneuver) on the fixed coupling map using SWAP moves. We FIRST
verify exact functional equivalence to the target circuit (every required
interaction applied exactly once, on physically-adjacent slots, no extras),
THEN count the inserted SWAP gates (fewer = better).

Score (minimization): sc = min(1000, 100 * B / F), ratio = sc/1000,
where B is the checker's own greedy in-order routing baseline. Any feasibility
or equivalence violation -> Ratio: 0.0.
"""
import sys
from collections import deque


def shortest_path(adj, s, t):
    if s == t:
        return [s]
    prev = {s: None}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in sorted(adj[u]):
            if v not in prev:
                prev[v] = u
                if v == t:
                    path = [t]
                    while path[-1] != s:
                        path.append(prev[path[-1]])
                    return path[::-1]
                q.append(v)
    return None


def baseline_swaps(n, adj, init, pairs):
    pos = list(init)
    inv = [0] * n
    for p in range(n):
        inv[pos[p]] = p
    sw = 0
    for (a, b) in pairs:
        path = shortest_path(adj, inv[a], inv[b])
        for i in range(len(path) - 2):
            u = path[i]
            v = path[i + 1]
            sw += 1
            lu = pos[u]
            lv = pos[v]
            pos[u] = lv
            pos[v] = lu
            inv[lv] = u
            inv[lu] = v
    return sw


def fail(reason):
    print("VIOLATION: %s  Ratio: 0.0" % reason)
    sys.exit(0)


def main():
    inst = open(sys.argv[1]).read().split()
    it = iter(inst)

    def ni():
        return int(next(it))

    n = ni()
    m = ni()
    k = ni()
    adj = [set() for _ in range(n)]
    edge_set = set()
    for _ in range(m):
        u = ni()
        v = ni()
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    init = [ni() for _ in range(n)]
    pairs = [(ni(), ni()) for _ in range(k)]

    # ---- parse participant program ----
    raw = open(sys.argv[2]).read().split("\n")
    prog = []
    for line in raw:
        s = line.strip()
        if not s:
            continue
        tok = s.split()
        op = tok[0].upper()
        if op == "SWAP":
            if len(tok) != 3:
                fail("bad SWAP arity")
            try:
                p = int(tok[1]); q = int(tok[2])
            except ValueError:
                fail("non-integer SWAP operand")
            prog.append(("SWAP", p, q))
        elif op == "GATE":
            if len(tok) != 2:
                fail("bad GATE arity")
            try:
                g = int(tok[1])
            except ValueError:
                fail("non-integer GATE operand")
            prog.append(("GATE", g, 0))
        else:
            fail("unknown instruction '%s'" % tok[0])

    # sanity cap: program length cannot exceed a generous bound
    if len(prog) > 200000:
        fail("program too long")

    # ---- simulate ----
    pos = list(init)
    inv = [0] * n
    for p in range(n):
        inv[pos[p]] = p
    applied = [0] * k
    swaps = 0
    for instr in prog:
        if instr[0] == "SWAP":
            p, q = instr[1], instr[2]
            if not (0 <= p < n and 0 <= q < n):
                fail("SWAP slot out of range")
            if p == q:
                fail("SWAP on identical slot")
            if (min(p, q), max(p, q)) not in edge_set:
                fail("SWAP on non-adjacent slots (%d,%d)" % (p, q))
            lu = pos[p]; lv = pos[q]
            pos[p] = lv; pos[q] = lu
            inv[lv] = p; inv[lu] = q
            swaps += 1
        else:
            g = instr[1]
            if not (0 <= g < k):
                fail("GATE index out of range")
            a, b = pairs[g]
            pa = inv[a]; pb = inv[b]
            if (min(pa, pb), max(pa, pb)) not in edge_set:
                fail("interaction %d executed on non-adjacent slots" % g)
            applied[g] += 1
            if applied[g] > 1:
                fail("interaction %d executed more than once (breaks equivalence)" % g)

    for g in range(k):
        if applied[g] != 1:
            fail("interaction %d not executed exactly once" % g)

    F = swaps
    B = baseline_swaps(n, adj, init, pairs)
    if B <= 0:
        # degenerate instance safeguard (gen guarantees B>0)
        B = 1
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("swaps=%d baseline=%d  Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
