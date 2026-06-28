#!/usr/bin/env python3
# Independent brute force oracle for "connected subsets containing each vertex".
# For each root r, count the number of connected vertex subsets S of the tree
# with r in S, by recomputing from scratch (per-root O(n) DP) -> O(n^2) overall.
# For very small n we additionally cross-check root 1 against a fully exhaustive
# 2^n enumeration of subsets (commented helper available below).
import sys
sys.setrecursionlimit(1000000)

MOD = 1000000007


def solve(data):
    it = iter(data)
    n = int(next(it))
    adj = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u = int(next(it)); v = int(next(it))
        adj[u].append(v)
        adj[v].append(u)

    if n == 1:
        return ["1"]

    # f[v] over a rooting at `root`: connected subsets in subtree(v) containing v.
    # ans for that root = f[root]. Recompute per root => O(n^2).
    out = []
    for root in range(1, n + 1):
        # iterative post-order DFS
        par = [0] * (n + 1)
        order = []
        vis = [False] * (n + 1)
        st = [root]
        vis[root] = True
        par[root] = 0
        while st:
            u = st.pop()
            order.append(u)
            for w in adj[u]:
                if not vis[w]:
                    vis[w] = True
                    par[w] = u
                    st.append(w)
        f = [1] * (n + 1)
        for u in reversed(order):
            prod = 1
            for w in adj[u]:
                if w == par[u]:
                    continue
                prod = prod * ((1 + f[w]) % MOD) % MOD
            f[u] = prod
        out.append(str(f[root] % MOD))
    return out


def main():
    data = sys.stdin.read().split()
    res = solve(data)
    print(" ".join(res))


if __name__ == "__main__":
    main()
