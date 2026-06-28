#!/usr/bin/env python3
"""
TRULY independent oracle (exponential): for very small n, try every subset of
non-important vertices to delete and find the minimum-size subset that
disconnects all important vertices. Used only to validate the DP rule itself.
Same I/O format as sol.cpp.
"""
import sys
from itertools import combinations


def connected_components_with_important(n, adj, deleted, imp):
    """Return True if no two important vertices are in the same component after
    removing 'deleted' vertices."""
    seen = [False] * (n + 1)
    for start in range(1, n + 1):
        if seen[start] or start in deleted:
            continue
        # BFS this component
        comp_imp = 0
        stack = [start]
        seen[start] = True
        while stack:
            u = stack.pop()
            if imp[u]:
                comp_imp += 1
                if comp_imp >= 2:
                    return False
            for w in adj[u]:
                if not seen[w] and w not in deleted:
                    seen[w] = True
                    stack.append(w)
    return True


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
    q = int(data[idx]); idx += 1
    out = []
    for _ in range(q):
        k = int(data[idx]); idx += 1
        S = [int(x) for x in data[idx:idx + k]]
        idx += k
        imp = [False] * (n + 1)
        for x in S:
            imp[x] = True
        impset = set(S)
        # adjacency among important -> impossible
        bad = any(any(imp[w] for w in adj[v]) for v in S)
        if bad:
            out.append("-1")
            continue
        candidates = [v for v in range(1, n + 1) if v not in impset]
        best = None
        # try increasing sizes
        for size in range(0, len(candidates) + 1):
            found = False
            for combo in combinations(candidates, size):
                if connected_components_with_important(n, adj, set(combo), imp):
                    best = size
                    found = True
                    break
            if found:
                break
        out.append(str(best))
    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
