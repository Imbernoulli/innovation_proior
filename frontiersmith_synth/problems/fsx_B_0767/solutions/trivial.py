# TIER: trivial
"""Naive per-edge stitching: visit every allowed window in a fixed (sorted, structure-
blind) order, walking back-and-forth via BFS from wherever the tour currently sits. This
reproduces the checker's own internal baseline construction almost exactly -- it never
tries to batch nearby edges together, so it pays full round-trip cost for every edge that
isn't already adjacent to the current position."""
import sys, string
from collections import deque

DIGITS = string.digits


def node_list(k, nm1):
    if nm1 == 0:
        return [""]
    out = [""]
    for _ in range(nm1):
        out = [p + d for p in out for d in DIGITS[:k]]
    return sorted(out)


def full_edges(k, L):
    return sorted(p + d for p in node_list(k, L - 1) for d in DIGITS[:k])


def build_out_adj(allowed_sorted):
    adj = {}
    for w in allowed_sorted:
        u, v = w[:-1], w[1:]
        adj.setdefault(u, []).append((v, w[-1]))
    for u in adj:
        adj[u].sort()
    return adj


def bfs_path_chars(src, dst, out_adj):
    if src == dst:
        return []
    prev = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            break
        for v, c in out_adj.get(u, []):
            if v not in prev:
                prev[v] = (u, c)
                q.append(v)
    if dst not in prev:
        return None
    chars = []
    cur = dst
    while cur != src:
        pu, pc = prev[cur]
        chars.append(pc)
        cur = pu
    chars.reverse()
    return chars


def main():
    toks = sys.stdin.read().split()
    k, L, m = int(toks[0]), int(toks[1]), int(toks[2])
    forbidden = set(toks[3:3 + m])
    full = full_edges(k, L)
    allowed_sorted = sorted(set(full) - forbidden)
    out_adj = build_out_adj(allowed_sorted)
    act = sorted({w[:-1] for w in allowed_sorted} | {w[1:] for w in allowed_sorted})
    start = act[0]
    pos = start
    chars = []
    for w in allowed_sorted:
        u = w[:-1]
        if pos != u:
            path = bfs_path_chars(pos, u, out_adj)
            chars.extend(path)
            pos = u
        chars.append(w[-1])
        pos = w[1:]
    if pos != start:
        path = bfs_path_chars(pos, start, out_adj)
        chars.extend(path)
    print("".join(chars))


if __name__ == "__main__":
    main()
