# TIER: strong
"""The insight: a cyclic string covering every allowed window exactly corresponds to a
closed walk in the de Bruijn graph on (L-1)-mers (edges = length-L windows) that uses every
allowed edge at least once, never using a forbidden edge. If that graph were balanced
(in-degree == out-degree everywhere) and connected, the *shortest* such walk is simply an
Eulerian circuit -- length exactly the number of allowed edges, found by Hierholzer's
algorithm. Deleting the forbidden edges generally breaks the balance. So before tracing any
tour we first REPAIR it: compute each node's (in-degree - out-degree) imbalance, and for
every node with excess in-degree ("needs an extra out-edge") route the shortest possible
allowed path to some node with excess out-degree ("needs an extra in-edge"), splicing that
path's edges into the graph as extra copies. This restores exact balance with (close to)
minimum extra length, using only real allowed edges -- a global dual/repair view, not a
local append-the-next-symbol search. Only then do we run Hierholzer's algorithm to read off
the actual tour."""
import sys, string
from collections import deque

DIGITS = string.digits


def node_list(k, nm1):
    if nm1 == 0:
        return [""]
    out = [""]
    for _ in range(nm1):
        out = [p + d for p in out for d in DIGITS[:k]]
    return sorted(out)


def full_edges(k, L):
    return sorted(p + d for p in node_list(k, L - 1) for d in DIGITS[:k])


def build_out_adj(allowed_sorted):
    adj = {}
    for w in allowed_sorted:
        u, v = w[:-1], w[1:]
        adj.setdefault(u, []).append((v, w[-1]))
    for u in adj:
        adj[u].sort()
    return adj


def imbalance(allowed_sorted):
    ind, outd = {}, {}
    for w in allowed_sorted:
        u, v = w[:-1], w[1:]
        outd[u] = outd.get(u, 0) + 1
        ind[v] = ind.get(v, 0) + 1
    nodes = set(ind) | set(outd)
    return {n: ind.get(n, 0) - outd.get(n, 0) for n in nodes}


def balance_via_shortest_paths(adj, allowed_sorted):
    """Mutates `adj` in place, appending augmenting-path edges so every node becomes
    balanced. Shortest paths are computed on the ORIGINAL allowed graph (adj as given),
    each plus-unit matched greedily to its nearest still-open minus-unit (not a globally
    optimal assignment -- a genuine but not exhaustive repair)."""
    d = imbalance(allowed_sorted)
    plus = sorted(n for n, v in d.items() if v > 0 for _ in range(v))
    minus = sorted(n for n, v in d.items() if v < 0 for _ in range(-v))
    used_minus = [False] * len(minus)
    for p in plus:
        dist = {p: 0}
        prev = {p: None}
        q = deque([p])
        while q:
            u = q.popleft()
            for v, c in adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    prev[v] = (u, c)
                    q.append(v)
        best_j, best_d = -1, None
        for j, mnode in enumerate(minus):
            if used_minus[j]:
                continue
            dd = dist.get(mnode)
            if dd is None:
                continue
            if best_d is None or dd < best_d or (dd == best_d and mnode < minus[best_j]):
                best_d, best_j = dd, j
        if best_j == -1:
            continue  # instance guarantees strong connectivity, so this should not happen
        used_minus[best_j] = True
        mnode = minus[best_j]
        edges = []
        cur = mnode
        while cur != p:
            pu, pc = prev[cur]
            edges.append((pu, pc))
            cur = pu
        edges.reverse()
        for (u, c) in edges:
            adj.setdefault(u, []).append((adj_target(u, c), c))


def adj_target(u, c):
    return u[1:] + c


def euler_circuit_chars(start, adj):
    ptr = {}
    stack = [start]
    charstack = []
    result = []
    while stack:
        v = stack[-1]
        lst = adj.get(v, [])
        i = ptr.get(v, 0)
        if i < len(lst):
            nxt, c = lst[i]
            ptr[v] = i + 1
            stack.append(nxt)
            charstack.append(c)
        else:
            stack.pop()
            if charstack:
                result.append(charstack.pop())
    result.reverse()
    return result


def main():
    toks = sys.stdin.read().split()
    k, L, m = int(toks[0]), int(toks[1]), int(toks[2])
    forbidden = set(toks[3:3 + m])
    full = full_edges(k, L)
    allowed_sorted = sorted(set(full) - forbidden)
    adj = build_out_adj(allowed_sorted)
    act = sorted({w[:-1] for w in allowed_sorted} | {w[1:] for w in allowed_sorted})
    start = act[0]

    balance_via_shortest_paths(adj, allowed_sorted)

    result_chars = euler_circuit_chars(start, adj)
    print("".join(result_chars))


if __name__ == "__main__":
    main()
