# TIER: greedy
"""Textbook spanning-tree construction: scan the given fusion edges in the
order they appear in the input and keep an edge whenever it does not close a
cycle (a plain Kruskal-style spanning forest, taking the FIRST candidate
that works at every junction). This is the "just pick a spanning tree" first
instinct -- it never looks at where facets actually land, so it can (and on
several tests does) fold two facets on top of each other."""
import sys


def main():
    toks = sys.stdin.read().split()
    n = int(toks[0]); m = int(toks[1])
    idx = 3
    edges = []
    for _ in range(m):
        u = int(toks[idx]); su = int(toks[idx + 1])
        v = int(toks[idx + 2]); sv = int(toks[idx + 3])
        idx += 4
        edges.append((u, su, v, sv))

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    kept = []
    for i, (u, su, v, sv) in enumerate(edges):
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
            kept.append(i)

    print(len(kept))
    print(" ".join(map(str, kept)))


if __name__ == "__main__":
    main()
