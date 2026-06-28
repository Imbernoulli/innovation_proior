#!/usr/bin/env python3
"""
Brute-force oracle for "Kingdom and its Cities" (virtual-tree problem).

Reads the SAME input format as sol.cpp from stdin, but solves each query by a
plain full-tree DP rooted at vertex 1 (the slow O(n) per query approach, total
O(q*n)). This is the obviously-correct reference; it only needs n <= ~500.

Format:
  n
  n-1 lines: u v   (tree edges, 1-indexed)
  q
  q queries: first integer k, then k distinct important vertices

Per query, output the minimum number of NON-important vertices to delete so that
no two important vertices remain connected, or -1 if impossible (two important
vertices are adjacent).
"""
import sys
from sys import setrecursionlimit


def main():
    data = sys.stdin.buffer.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    adj = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        adj[u].append(v)
        adj[v].append(u)

    # Root the tree at 1, compute parent + a post-order (children processed
    # before parents). Iterative to avoid recursion limits.
    parent = [0] * (n + 1)
    order = []
    visited = [False] * (n + 1)
    stack = [1]
    visited[1] = True
    parent[1] = 0
    while stack:
        u = stack.pop()
        order.append(u)
        for w in adj[u]:
            if not visited[w]:
                visited[w] = True
                parent[w] = u
                stack.append(w)
    # order is a pre-order (parent before child); reverse for post-order.
    post = order[::-1]

    q = int(data[idx]); idx += 1
    out = []
    for _ in range(q):
        k = int(data[idx]); idx += 1
        S = data[idx:idx + k]
        idx += k
        imp = [False] * (n + 1)
        for x in S:
            imp[int(x)] = True

        # Impossible iff two important vertices are adjacent (here: an important
        # vertex whose parent is important).
        bad = False
        for x in S:
            v = int(x)
            if v != 1 and imp[parent[v]]:
                bad = True
                break
        if bad:
            out.append("-1")
            continue

        # Full-tree DP.
        # cnt[v] = number of important vertices in v's subtree that are still
        #          connected up to v (path from them to v not yet cut).
        # ans = deletions made so far.
        cnt = [0] * (n + 1)
        ans = 0
        for v in post:
            s = 0
            for w in adj[v]:
                if w != parent[v]:
                    s += cnt[w]
            if imp[v]:
                # v itself is important and is a "still-connected" vertex.
                # Every child subtree that still carries a connected important
                # vertex (cnt[w] > 0) must be severed: one deletion each. The
                # cheapest cut is to delete the child-side junction, but for the
                # COUNT it is exactly the number of children with cnt[w] > 0.
                for w in adj[v]:
                    if w != parent[v] and cnt[w] > 0:
                        ans += 1
                cnt[v] = 1
            else:
                if s >= 2:
                    # Two or more connected important vertices meet at v; delete
                    # v itself to separate all of them at once.
                    ans += 1
                    cnt[v] = 0
                elif s == 1:
                    cnt[v] = 1  # pass the single connected vertex upward
                else:
                    cnt[v] = 0
        out.append(str(ans))

    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    setrecursionlimit(1000000)
    main()
