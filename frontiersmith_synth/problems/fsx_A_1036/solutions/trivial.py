# TIER: trivial
"""Reproduces the checker's own baseline: pick the single heaviest DFA that has SOME
reachable accepting string within Lmax steps, and output only that DFA's shortest
accepting string (lexicographically smallest among ties). No attempt to combine
DFAs at all."""
import sys
from collections import deque


def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)
    def nxt():
        return int(next(it))
    m = nxt(); A = nxt(); Lmax = nxt()
    dfas = []
    for _ in range(m):
        n = nxt(); start = nxt(); w = nxt()
        k = nxt()
        accept = set(nxt() for _ in range(k))
        trans = [[nxt() for _ in range(A)] for _ in range(n)]
        dfas.append({"n": n, "start": start, "w": w, "accept": accept, "trans": trans})
    return m, A, Lmax, dfas


def bfs_path(dfa, Lmax, A):
    start = dfa["start"]
    if start in dfa["accept"]:
        return []
    dist = {start: 0}
    parent = {}
    q = deque([start])
    while q:
        s = q.popleft()
        d = dist[s]
        if d >= Lmax:
            continue
        for a in range(A):
            ns = dfa["trans"][s][a]
            if ns not in dist:
                dist[ns] = d + 1
                parent[ns] = (s, a)
                if ns in dfa["accept"]:
                    path = []
                    cur = ns
                    while cur != start:
                        p, sym = parent[cur]
                        path.append(sym)
                        cur = p
                    path.reverse()
                    return path
                q.append(ns)
    return None


def main():
    m, A, Lmax, dfas = read_instance()
    best_w, best_path = -1, []
    for d in dfas:
        p = bfs_path(d, Lmax, A)
        if p is not None and d["w"] > best_w:
            best_w, best_path = d["w"], p
    print(len(best_path))
    print(" ".join(map(str, best_path)))


if __name__ == "__main__":
    main()
