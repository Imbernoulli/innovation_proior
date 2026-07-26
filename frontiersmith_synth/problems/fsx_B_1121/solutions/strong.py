# TIER: strong
"""The insight: which vertex serves as the synchronizing sink is a FREE
DESIGN CHOICE, not a given. Instead of committing to vertex 0, search every
candidate vertex that has a self-loop among its own slots, build the
shortest spanning in-tree toward EACH of them (reverse-BFS, same primitive
as greedy), and keep the root whose tree has the smallest depth. Aligning
the coloring's compressing symbol with THAT in-tree is what the road-
coloring theorem actually promises: a deliberately chosen tree, not an
arbitrary/default one, is what makes the reset word short. This is a
reformulation of the task (search over roots) rather than a tuned version of
the greedy recipe."""
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

    best = None  # (max_depth, root, slot_choice)
    for r in range(n):
        tree = build_tree(n, targets, r)
        if tree is None:
            continue
        depth, slot_choice = tree
        d = max(depth)
        if best is None or d < best[0]:
            best = (d, r, slot_choice)

    if best is None:
        coloring = [list(range(m)) for _ in range(n)]
    else:
        _, _, slot_choice = best
        coloring = make_coloring(n, m, slot_choice)

    out = []
    for v in range(n):
        out.append(" ".join(str(x) for x in coloring[v]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
