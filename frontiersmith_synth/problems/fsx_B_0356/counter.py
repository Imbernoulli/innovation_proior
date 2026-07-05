#!/usr/bin/env python3
"""counter.py <in> <out> <ans>  (ans ignored)  -- Format D op-count scorer.

Verifies EXACT functional equivalence of the submitted SWAP schedule to the target
QAOA cost circuit (every required ZZ interaction applied exactly the required number
of times, each while its two lots occupy adjacent slots, every SWAP a real coupling
edge), THEN counts the number of SWAP moves used. Fewer moves -> higher score.

    ratio = min(1.0, 0.1 * B / F)      # F = participant SWAP count, B = internal baseline

Baseline B = a route-and-undo construction the checker builds itself (bring each
interacting pair together along a shortest path, apply, then undo). trivial reproduces
this exactly -> ~0.1; sharing state / reordering can do much better.

ANY feasibility violation, non-integer token, or incomplete circuit -> Ratio 0.0.
"""
import sys
from collections import deque, Counter


def die0(reason):
    print("infeasible: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)

    def nxt():
        return int(next(it))

    Q = nxt(); E = nxt(); K = nxt()
    edges = set()
    adj = [[] for _ in range(Q)]
    for _ in range(E):
        u = nxt(); v = nxt()
        k = (u, v) if u < v else (v, u)
        if k not in edges:
            edges.add(k)
            adj[u].append(v)
            adj[v].append(u)
    inter = []
    for _ in range(K):
        a = nxt(); b = nxt()
        inter.append((a, b))
    for a in range(Q):
        adj[a].sort()
    return Q, edges, adj, inter


def bfs_dist(adj, Q, src, dst):
    if src == dst:
        return 0
    seen = [False] * Q
    seen[src] = True
    dq = deque([(src, 0)])
    while dq:
        node, d = dq.popleft()
        for nb in adj[node]:
            if not seen[nb]:
                if nb == dst:
                    return d + 1
                seen[nb] = True
                dq.append((nb, d + 1))
    return -1  # disconnected (should not happen)


def main():
    inpath, outpath = sys.argv[1], sys.argv[2]
    Q, edges, adj, inter = read_instance(inpath)

    required = Counter()
    for (a, b) in inter:
        required[(a, b) if a < b else (b, a)] += 1

    # ---- internal baseline B: route-and-undo on the fixed identity placement ----
    B = 0
    for (a, b) in inter:
        d = bfs_dist(adj, Q, a, b)  # identity: logical i at physical i
        if d < 0:
            die0("checker: coupling graph disconnected")
        B += 2 * (d - 1)  # d==1 -> 0 swaps

    # ---- simulate the participant schedule ----
    try:
        with open(outpath) as f:
            raw = f.read()
    except Exception:
        die0("cannot read output")

    pos = list(range(Q))   # pos[logical] = physical
    occ = list(range(Q))   # occ[physical] = logical
    remaining = Counter(required)
    F = 0
    n_lines = 0
    LIMIT = 50 * (len(inter) + Q * Q) + 1000

    for line in raw.splitlines():
        parts = line.split()
        if not parts:
            continue
        n_lines += 1
        if n_lines > LIMIT:
            die0("too many instructions")
        op = parts[0]
        if op == "S":
            if len(parts) != 3:
                die0("bad SWAP arity")
            try:
                p = int(parts[1]); q = int(parts[2])
            except ValueError:
                die0("non-integer SWAP operand")
            if not (0 <= p < Q and 0 <= q < Q) or p == q:
                die0("SWAP out of range / self")
            key = (p, q) if p < q else (q, p)
            if key not in edges:
                die0("SWAP on non-coupling pair")
            la, lb = occ[p], occ[q]
            occ[p], occ[q] = lb, la
            pos[la], pos[lb] = q, p
            F += 1
        elif op == "G":
            if len(parts) != 3:
                die0("bad interaction arity")
            try:
                a = int(parts[1]); b = int(parts[2])
            except ValueError:
                die0("non-integer interaction operand")
            if not (0 <= a < Q and 0 <= b < Q) or a == b:
                die0("interaction out of range / self")
            key = (a, b) if a < b else (b, a)
            if remaining.get(key, 0) <= 0:
                die0("interaction not required / applied too many times")
            pa, pb = pos[a], pos[b]
            ek = (pa, pb) if pa < pb else (pb, pa)
            if ek not in edges:
                die0("interaction on non-adjacent slots")
            remaining[key] -= 1
        else:
            die0("unknown instruction '%s'" % op[:8])

    for k, v in remaining.items():
        if v != 0:
            die0("circuit incomplete: missing required interaction(s)")

    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    print("F(swaps)=%d B(baseline)=%d" % (F, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
