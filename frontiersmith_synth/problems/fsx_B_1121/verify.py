#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- deterministic checker for road-coloring-synchronize.

Reads the instance (n, m, per-vertex edge slots) from <in>, the participant's
edge-labeling from <out>. Validates that the labeling is, at every vertex, a
bijection from {0..m-1} to that vertex's m slots (a valid "coloring"). Builds
the resulting deterministic transition function and computes the EXACT length
of the shortest word that resets (synchronizes) the automaton via breadth-
first search over the power-automaton (subset) graph -- a fully deterministic,
O(size) computation for the small n used here. A non-synchronizing coloring
scores 0.0, same as any other feasibility violation.

The checker also builds its OWN reference coloring (root vertex 0, the
"default chain" only) and scores that with the identical BFS to obtain a
positive internal baseline B. Objective is minimization, so
    ratio = min(1000, 100*B/F) / 1000
shorter reset words score higher; ratio saturates at 1.0 only if the
submission is 10x shorter than the baseline (headroom is intentional: see
strong.py, which searches ALL candidate roots and typically lands well shy
of that saturation point).
"""
import sys
from collections import deque


def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    targets = []
    for _ in range(n):
        targets.append([int(next(it)) for _ in range(m)])
    return n, m, targets


def build_tree(n, targets, root):
    """BFS a shortest spanning in-tree toward `root` using only edges already
    present in `targets`. Returns (depth_list, slot_choice_list) or None if
    `root` has no self-loop among its own slots or the tree cannot span all
    n vertices."""
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


def reset_word_length(n, m, delta):
    """Exact shortest synchronizing-word length via BFS on the power automaton.
    delta[v][a] = target state. Returns int length, or None if not
    synchronizing (no reachable subset ever collapses to size 1)."""
    start = frozenset(range(n))
    if len(start) == 1:
        return 0
    seen = {start: 0}
    dq = deque([start])
    while dq:
        cur = dq.popleft()
        d = seen[cur]
        for a in range(m):
            nxt = frozenset(delta[v][a] for v in cur)
            if nxt not in seen:
                seen[nxt] = d + 1
                if len(nxt) == 1:
                    return d + 1
                dq.append(nxt)
    return None


def coloring_to_delta(n, m, targets, coloring):
    delta = [[None] * m for _ in range(n)]
    for v in range(n):
        for i in range(m):
            a = coloring[v][i]
            delta[v][a] = targets[v][i]
    return delta


def fail(msg):
    print(f"INFEASIBLE: {msg} Ratio: 0.0")
    sys.exit(0)


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]
    n, m, targets = read_instance(in_path)

    try:
        out_toks = open(out_path).read().split()
    except Exception:
        fail("cannot read output")
        return

    if len(out_toks) != n * m:
        fail(f"expected {n*m} integers, got {len(out_toks)}")
        return

    parsed = []
    for tok in out_toks:
        try:
            v = int(tok)
        except ValueError:
            fail(f"non-integer token {tok!r}")
            return
        if v != v or v in (float("inf"), float("-inf")):
            fail("non-finite token")
            return
        parsed.append(v)

    coloring = [parsed[v * m:(v + 1) * m] for v in range(n)]
    for v in range(n):
        if sorted(coloring[v]) != list(range(m)):
            fail(f"vertex {v} labels are not a permutation of 0..{m-1}: {coloring[v]}")
            return

    delta = coloring_to_delta(n, m, targets, coloring)
    F = reset_word_length(n, m, delta)
    if F is None:
        fail("resulting automaton is NOT synchronizing (no word collapses all states)")
        return

    base_tree = build_tree(n, targets, 0)
    if base_tree is None:
        # generator guarantees vertex 0 always has a self-loop and the chain
        # always spans everything; this branch should be unreachable.
        B = float(n * n)
    else:
        _, base_slot_choice = base_tree
        base_coloring = make_coloring(n, m, base_slot_choice)
        base_delta = coloring_to_delta(n, m, targets, base_coloring)
        base_F = reset_word_length(n, m, base_delta)
        B = float(base_F) if base_F is not None else float(n * n)
    B = max(B, 1e-9)

    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    ratio = sc / 1000.0
    print(f"n={n} m={m} F={F} B={B:.3f} Ratio: {ratio:.6f}")


if __name__ == "__main__":
    main()
