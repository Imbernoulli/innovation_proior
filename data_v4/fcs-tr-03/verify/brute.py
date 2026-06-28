#!/usr/bin/env python3
# Brute-force oracle for "subtree distinct colors".
# For each node, compute the set of distinct colors in its rooted subtree directly.
# Root is node 1 (1-indexed). O(n^2) worst case -> only for small n.
import sys
sys.setrecursionlimit(1 << 20)


def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    if n == 0:
        return
    color = [int(next(it)) for _ in range(n)]
    g = [[] for _ in range(n)]
    for _ in range(n - 1):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        g[u].append(v)
        g[v].append(u)

    # Build children lists rooted at 0 via iterative BFS.
    par = [-1] * n
    order = []
    vis = [False] * n
    stack = [0]
    vis[0] = True
    children = [[] for _ in range(n)]
    while stack:
        u = stack.pop()
        order.append(u)
        for w in g[u]:
            if not vis[w]:
                vis[w] = True
                par[w] = u
                children[u].append(w)
                stack.append(w)

    # Each node's subtree color set = union of children sets + own color.
    # Process in reverse topological order (children before parents).
    subtree_colors = [None] * n
    for u in reversed(order):
        s = set()
        s.add(color[u])
        for w in children[u]:
            s |= subtree_colors[w]
        subtree_colors[u] = s

    out = []
    for i in range(n):
        out.append(str(len(subtree_colors[i])))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
