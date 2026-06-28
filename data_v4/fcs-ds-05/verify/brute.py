#!/usr/bin/env python3
"""
Independent brute-force oracle for offline dynamic connectivity.

It does NOT use any clever data structure: it maintains the current edge set
literally as a Python set, and for every connectivity query it recomputes the
answer from scratch with a plain BFS over the present edges. O(q * (n + edges))
per run -- obviously correct, far too slow for the real limits, perfect oracle.
"""
import sys
from collections import deque, defaultdict


def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    q = int(data[idx]); idx += 1

    edges = set()              # set of frozenset({u, v})
    out = []

    for _ in range(q):
        typ = int(data[idx]); idx += 1
        u = int(data[idx]); idx += 1
        v = int(data[idx]); idx += 1
        if typ == 1:
            edges.add((min(u, v), max(u, v)))
        elif typ == 2:
            edges.discard((min(u, v), max(u, v)))
        else:
            # BFS from u over the current edge set
            if u == v:
                out.append("YES")
                continue
            adj = defaultdict(list)
            for (a, b) in edges:
                adj[a].append(b)
                adj[b].append(a)
            seen = {u}
            dq = deque([u])
            found = False
            while dq:
                x = dq.popleft()
                if x == v:
                    found = True
                    break
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y)
                        dq.append(y)
            out.append("YES" if found else "NO")

    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
