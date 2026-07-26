#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE road-coloring-synchronize instance to stdout.

Instance: a directed multigraph on n vertices, every vertex has exactly m
outgoing "slots" (physical road segments). gen.py plants two structural
features, invisible to the solver except through the raw edge lists:

  * a "default chain": edge v -> v-1 for v=1..n-1, and a self-loop 0 -> 0.
    Rooting a spanning in-tree at vertex 0 using ONLY this chain always
    works but has worst-case depth n-1 (a long reset word).
  * a "shortcut hub" gr2 (a uniformly random vertex != 0): gr2 gets its
    own self-loop, and most other vertices get a direct edge straight to
    gr2. Rooting a spanning in-tree at gr2 instead typically collapses in
    2-3 rounds -- a MUCH shorter reset word -- but finding this requires
    searching over candidate roots rather than defaulting to vertex 0.

All randomness is seeded deterministically from testId only.
"""
import sys
import random
from collections import deque


SIZES = [6, 7, 7, 8, 8, 9, 9, 10, 10, 11]


# ---- private helpers used ONLY to steer generation toward genuine traps;
# never shown to the solver, and structurally identical to the public
# checker's own synchronization search (verify.py) so "bad for identity
# labeling" is verified exactly, not guessed at. ----
def _build_tree(n, targets, root):
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


def _reset_word_length(n, m, delta):
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


def _identity_F(n, m, targets):
    delta = [[None] * m for _ in range(n)]
    for v in range(n):
        for i in range(m):
            delta[v][i] = targets[v][i]
    return _reset_word_length(n, m, delta)


def _baseline_B(n, m, targets):
    tree = _build_tree(n, targets, 0)
    if tree is None:
        return None
    _, slot_choice = tree
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
    delta = [[None] * m for _ in range(n)]
    for v in range(n):
        for i in range(m):
            a = coloring[v][i]
            delta[v][a] = targets[v][i]
    return _reset_word_length(n, m, delta)


def gen(test_id):
    base_rng = random.Random(1000003 * test_id + 7)
    n = SIZES[(test_id - 1) % len(SIZES)]
    m = 4
    # far-half hub position: keeps the default chain from vertex 0 genuinely
    # long (the hub does not rescue it -- reaching the hub via the chain is
    # itself already most of the way to n-1), while rooting at the hub
    # collapses everyone else in one hop.
    lo = max(1, n // 2)
    gr2 = base_rng.randint(lo, n - 1)
    noise = [base_rng.randint(0, n - 1) for _ in range(n)]

    roles = []
    for v in range(n):
        chain_t = 0 if v == 0 else v - 1   # -> forms the long default chain into 0
        ring_t = (v + 1) % n                # -> guarantees strong connectivity
        hub_t = gr2                         # -> every vertex, one hop from the hub
        noise_t = noise[v]                  # -> uninformative distractor edge
        roles.append([chain_t, ring_t, hub_t, noise_t])

    # Search over SLOT-ORDER shuffles only (never the underlying edges/roles,
    # so greedy/strong -- which read targets, not slot positions -- are
    # unaffected) for one under which the naive "identity" labeling (symbol i
    # -> slot i, the trivial-tier recipe) performs badly: non-synchronizing,
    # or at least clearly worse than the checker's own root-0 baseline. This
    # deterministically PLANTS the trap the addendum requires, verified by
    # exact simulation rather than left to chance.
    shuf_rng = random.Random(97 * test_id + 13)
    best_edges, best_score = None, None  # score: lower is a better (worse-for-trivial) case
    B = _baseline_B(n, m, roles)
    for _attempt in range(400):
        cand = [list(r) for r in roles]
        for v in range(n):
            shuf_rng.shuffle(cand[v])
        F = _identity_F(n, m, cand)
        score = 10 ** 9 if F is None else F  # non-synchronizing is the best trap
        if F is not None and B is not None and F <= B:
            continue  # identity ties/beats the baseline -- not a trap, keep searching
        if best_score is None or score > best_score:
            best_edges, best_score = cand, score
        if F is None:
            break  # can't do better than genuinely non-synchronizing
    edges = best_edges if best_edges is not None else roles
    return n, m, edges


def main():
    test_id = int(sys.argv[1])
    n, m, edges = gen(test_id)
    out = [f"{n} {m}"]
    for v in range(n):
        out.append(" ".join(str(t) for t in edges[v]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
