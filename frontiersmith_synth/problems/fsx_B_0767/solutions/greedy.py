# TIER: greedy
"""The obvious first attempt: walk forward, always taking an outgoing edge whose window
hasn't been used yet (smallest symbol first). This is exactly Hierholzer's rule with NO
global planning. On a balanced+connected graph it usually closes up cleanly. But once
forbidden windows have knocked a graph out of balance, this single continuous walk can run
completely out of *unused* outgoing edges long before every window is covered -- it is then
forced to keep moving on already-used edges (repeats) until it happens to wander back near
a still-uncovered edge. We bound that blind wandering and, if it drags on, rescue it with a
purely LOCAL "walk to the nearest still-uncovered edge" patch -- unlike the strong solution,
this never anticipates the imbalance globally, so on badly imbalanced graphs it pays for
many small separate detours instead of one globally short repair."""
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


def bfs_to_nearest(src, targets, out_adj):
    """BFS from src until reaching any node in `targets`; returns (path_chars, node)."""
    if src in targets:
        return [], src
    prev = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        for v, c in out_adj.get(u, []):
            if v not in prev:
                prev[v] = (u, c)
                if v in targets:
                    chars = []
                    cur = v
                    while cur != src:
                        pu, pc = prev[cur]
                        chars.append(pc)
                        cur = pu
                    chars.reverse()
                    return chars, v
                q.append(v)
    return None, None


def main():
    toks = sys.stdin.read().split()
    k, L, m = int(toks[0]), int(toks[1]), int(toks[2])
    forbidden = set(toks[3:3 + m])
    full = full_edges(k, L)
    allowed_sorted = sorted(set(full) - forbidden)
    allowed_set = set(allowed_sorted)
    out_adj = build_out_adj(allowed_sorted)
    act = sorted({w[:-1] for w in allowed_sorted} | {w[1:] for w in allowed_sorted})
    start = act[0]

    used = set()
    pos = start
    chars = []
    hard_cap = 40 * len(allowed_set) + 2000
    steps = 0

    while len(used) < len(allowed_set) and steps < hard_cap:
        steps += 1
        outs = out_adj.get(pos, [])
        unused_here = [(v, c) for v, c in outs if (pos + c) not in used]
        if unused_here:
            v, c = unused_here[0]
            used.add(pos + c)
            chars.append(c)
            pos = v
            continue

        # stuck: no unused edge leaves the current node. The obvious fix a first-pass
        # coder reaches for is "just go patch whichever still-uncovered window comes
        # first in the input/scan order" -- resume at the lexicographically SMALLEST
        # still-uncovered source, not the nearest one. Unlike the strong solution (which
        # plans ALL repairs together as one globally-cheapest matching before moving at
        # all), this fixes deficiencies one at a time, in scan order, so on graphs with
        # several separate imbalance sites it happily re-crosses ground it already paid
        # to cross for an earlier, unrelated repair.
        remaining_sources = sorted({w[:-1] for w in allowed_set if w not in used})
        if not remaining_sources:
            break
        target = remaining_sources[0]
        path = bfs_path_chars(pos, target, out_adj)
        if path is None:
            break
        chars.extend(path)
        pos = target

    if pos != start:
        path = bfs_path_chars(pos, start, out_adj)
        if path is not None:
            chars.extend(path)

    print("".join(chars))


if __name__ == "__main__":
    main()
