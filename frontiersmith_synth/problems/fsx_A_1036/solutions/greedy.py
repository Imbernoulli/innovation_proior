# TIER: greedy
"""The obvious recipe: find each DFA's own shortest accepting string (BFS), then walk
DFAs in weight-descending order and greedily EXTEND a single shared candidate string
whenever the next DFA's requirement is consistent (one is a prefix of the other) with
what's already committed -- skip it otherwise. This "satisfy the heaviest DFA first,
then bolt on whatever else still fits" heuristic never reconsiders the initial choice,
so if the heaviest DFA's requirement conflicts with a GROUP of lighter DFAs that
together outweigh it, the group is abandoned one-by-one and never recovered."""
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
    paths = [bfs_path(d, Lmax, A) for d in dfas]
    order = sorted(range(m), key=lambda i: (-dfas[i]["w"], i))

    S = []
    for i in order:
        p = paths[i]
        if p is None:
            continue
        k = min(len(S), len(p))
        if S[:k] == p[:k]:
            if len(p) > len(S):
                S = list(p)
        # else: conflicts with what's already committed -> skip this DFA entirely

    print(len(S))
    print(" ".join(map(str, S)))


if __name__ == "__main__":
    main()
