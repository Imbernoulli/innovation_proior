# TIER: strong
"""Real matroid intersection via its transportation-network reduction (valid because
BOTH constraints here are partition matroids): build source -> scheme-A-class ->
item -> scheme-B-class -> sink, each item edge capacity 1, class edges capacity
cap1[c]/cap2[c]. Repeatedly BFS-find an augmenting path in the RESIDUAL graph and
push flow along it -- this residual-graph augmenting path IS the matroid exchange-
graph augmenting path (exchange-property + augmenting-path mechanism): each
augmentation grows the batch by exactly one, recovering exactly what the single-
pass greedy strands behind an early bridge pick.

When no augmenting path remains, the flow is maximum (= the batch is maximum), and
the max-flow/min-cut duality gives the certificate for free: let R = set of items
whose item-node is reachable from the source in the final residual graph, and take
A = the items NOT in R. Weak duality guarantees r1(A) + r2(complement of A) equals
the batch size for ANY feasible batch, and at THIS A it is tight (proven optimal)."""
import sys
from collections import deque


def main():
    data = sys.stdin.read().split()
    p = 0

    def nxt():
        nonlocal p
        v = int(data[p])
        p += 1
        return v

    n = nxt()
    K1 = nxt()
    K2 = nxt()
    col1 = [0] * n
    col2 = [0] * n
    for i in range(n):
        col1[i] = nxt() - 1
        col2[i] = nxt() - 1
    cap1 = [nxt() for _ in range(K1)]
    cap2 = [nxt() for _ in range(K2)]

    # ---- node layout: S=0, M1 classes, items, M2 classes, T ----
    S = 0
    M1base = 1
    ITbase = 1 + K1
    M2base = 1 + K1 + n
    T = 1 + K1 + n + K2
    N = T + 1

    cap = {}
    adj = [[] for _ in range(N)]

    def add(u, v, c):
        if (u, v) not in cap:
            adj[u].append(v)
            adj[v].append(u)
            cap[(u, v)] = 0
            cap[(v, u)] = 0
        cap[(u, v)] += c

    for c in range(K1):
        add(S, M1base + c, cap1[c])
    for i in range(n):
        add(M1base + col1[i], ITbase + i, 1)
        add(ITbase + i, M2base + col2[i], 1)
    for c in range(K2):
        add(M2base + c, T, cap2[c])

    def bfs_path():
        parent = {S: None}
        dq = deque([S])
        while dq:
            u = dq.popleft()
            if u == T:
                break
            for v in adj[u]:
                if v not in parent and cap[(u, v)] > 0:
                    parent[v] = u
                    dq.append(v)
        return parent if T in parent else None

    while True:
        parent = bfs_path()
        if parent is None:
            break
        v = T
        bottleneck = float("inf")
        while parent[v] is not None:
            u = parent[v]
            bottleneck = min(bottleneck, cap[(u, v)])
            v = u
        v = T
        while parent[v] is not None:
            u = parent[v]
            cap[(u, v)] -= bottleneck
            cap[(v, u)] += bottleneck
            v = u

    I = [i for i in range(n) if cap[(M1base + col1[i], ITbase + i)] == 0]

    # residual reachability from S -> certificate
    reach = {S}
    dq = deque([S])
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if v not in reach and cap[(u, v)] > 0:
                reach.add(v)
                dq.append(v)

    A = sorted(i for i in range(n) if (ITbase + i) not in reach)

    print(len(I))
    print(" ".join(str(i + 1) for i in I))
    print(1)
    print(str(len(A)) + (" " + " ".join(str(a + 1) for a in A) if A else ""))


if __name__ == "__main__":
    main()
