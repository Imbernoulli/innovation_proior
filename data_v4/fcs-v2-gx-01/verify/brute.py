#!/usr/bin/env python3
# Independent oracle: brute force over all subsets of edges.
# A subset is feasible iff it is a forest (graphic matroid) AND respects
# per-color capacities (partition matroid). Output the max feasible size.
#
# Forest test via union-find. O(2^m * m). Only used on tiny m.
import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    n = int(data[idx]); idx += 1
    m = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1
    cap = [0] * (K + 1)
    for c in range(1, K + 1):
        cap[c] = int(data[idx]); idx += 1
    eu = [0] * m
    ev = [0] * m
    ec = [0] * m
    for i in range(m):
        eu[i] = int(data[idx]); idx += 1
        ev[i] = int(data[idx]); idx += 1
        ec[i] = int(data[idx]); idx += 1

    best = 0
    for mask in range(1 << m):
        # color counts
        ok = True
        ccount = [0] * (K + 1)
        edges = []
        for i in range(m):
            if mask & (1 << i):
                ccount[ec[i]] += 1
                if ccount[ec[i]] > cap[ec[i]]:
                    ok = False
                    break
                edges.append(i)
        if not ok:
            continue
        # forest test with union-find
        parent = list(range(n + 1))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        isforest = True
        for i in edges:
            a = find(eu[i]); b = find(ev[i])
            if a == b:
                isforest = False
                break
            parent[a] = b
        if not isforest:
            continue
        cnt = len(edges)
        if cnt > best:
            best = cnt
    print(best)

if __name__ == "__main__":
    main()
