# TIER: greedy
"""The obvious textbook recipe: pick THE fixed vertex 0 as the synchronizing
sink, build a shortest spanning in-tree toward it using the given edges
(standard reverse-BFS), and align one symbol (0) with the tree so repeatedly
applying it funnels every state to vertex 0. This is a real application of
the road-coloring idea -- but it never considers whether some OTHER vertex
would make a dramatically shorter in-tree; it commits to the first/obvious
root and stops there."""
import sys
from collections import deque


def build_tree(n, targets, root):
    if not any(t == root for t in targets[root]):
        return None
    rev = [[] for _ in range(n)]
    for v in range(n):
        for t in targets[v]:
            rev[t].append(v)
    depth = [-1] * n
    parent_target = [None] * n
    depth[root] = 0
    dq = deque([root])
    while dq:
        t = dq.popleft()
        for v in rev[t]:
            if depth[v] == -1:
                depth[v] = depth[t] + 1
                parent_target[v] = t
                dq.append(v)
    if any(d == -1 for d in depth):
        return None
    slot_choice = [None] * n
    for v in range(n):
        want = root if v == root else parent_target[v]
        for i, t in enumerate(targets[v]):
            if t == want:
                slot_choice[v] = i
                break
    return depth, slot_choice


def make_coloring(n, m, slot_choice):
    coloring = []
    for v in range(n):
        c = [None] * m
        c[slot_choice[v]] = 0
        nxt = 1
        for i in range(m):
            if i == slot_choice[v]:
                continue
            c[i] = nxt
            nxt += 1
        coloring.append(c)
    return coloring


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    m = int(next(it))
    targets = [[int(next(it)) for _ in range(m)] for _ in range(n)]

    tree = build_tree(n, targets, 0)  # always try vertex 0, no search
    if tree is None:
        # fallback: identity (should not happen on generated instances)
        coloring = [list(range(m)) for _ in range(n)]
    else:
        _, slot_choice = tree
        coloring = make_coloring(n, m, slot_choice)

    out = []
    for v in range(n):
        out.append(" ".join(str(x) for x in coloring[v]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
