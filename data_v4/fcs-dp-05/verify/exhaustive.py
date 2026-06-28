#!/usr/bin/env python3
# Fully independent O(2^n) exhaustive oracle: enumerate every vertex subset,
# test connectivity by BFS within the induced subgraph, count per vertex.
# Used only for tiny n (n <= 14) to validate the per-root DP brute itself.
import sys

MOD = 1000000007


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    adj = [[] for _ in range(n + 1)]
    for _ in range(n - 1):
        u = int(next(it)); v = int(next(it))
        adj[u].append(v)
        adj[v].append(u)

    if n == 1:
        print("1")
        return

    cnt = [0] * (n + 1)
    for mask in range(1, 1 << n):
        verts = [i + 1 for i in range(n) if (mask >> i) & 1]
        s = set(verts)
        # connectivity check via BFS from verts[0]
        start = verts[0]
        seen = {start}
        stack = [start]
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y in s and y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(seen) == len(s):
            for v in verts:
                cnt[v] += 1
    print(" ".join(str(cnt[v] % MOD) for v in range(1, n + 1)))


if __name__ == "__main__":
    main()
