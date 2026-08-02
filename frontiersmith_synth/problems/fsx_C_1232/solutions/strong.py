# TIER: strong
"""
The insight: build the resource-DEPENDENCY GRAPH from the schema instead of
sampling endpoints blindly, then turn "how to spend the remaining budget"
into an explicit multiple-choice knapsack over per-type completion levels.

1. A type only yields score through its OWN local state chain (BFS shortest
   path from state 0 to its best-reachable state, using the transition
   table given in the input -- no assumption that "state index == depth"
   is hard-coded anywhere but the reward formula, which the checker also
   uses). Types with only state 0 (S_t == 1) are pure "scaffolding": zero
   direct value, but may gate other types via deps[].

2. For every type with S_t > 1 ("value type"), compute the ancestor
   closure (deps, transitively) it needs before it can even be CREATEd.
   In this schema family that closure is small and heavily SHARED across
   value types (a common prerequisite chain) -- so build the UNION of all
   such ancestors ONCE, up front, in topological order. Random/greedy
   sampling never recognizes this sharing and re-pays (or simply never
   completes) the dependency chain per attempt.

3. With the shared scaffold built, every remaining value type is
   independently completable at one of a few discrete "investment levels"
   j = 0..k (create + first j ops of its shortest path), each with a known
   (cost, value). Because the per-step reward along a chain is convex
   (state^2), finishing fewer chains fully is worth more than touching many
   chains partially -- this is exactly a 0/1 multiple-choice knapsack over
   remaining budget, solved here EXACTLY with a small DP (not "greedy plus
   more iterations").
"""
import sys
from collections import deque


def read_instance():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it)); C = int(next(it)); budget = int(next(it))
    deps = [None] * T
    tables = [None] * T
    for t in range(T):
        k = int(next(it))
        d = [int(next(it)) for _ in range(k)]
        S = int(next(it))
        table = []
        for s in range(S):
            table.append([int(next(it)) for _ in range(C)])
        deps[t] = d
        tables[t] = table
    return T, C, budget, deps, tables


def bfs_best_path(table):
    """Shortest-path (in opcodes) from state 0 to the highest-index
    reachable state. Returns list of (state_reached, opcode) in order."""
    S = len(table)
    C = len(table[0]) if S else 0
    dist = [-1] * S
    prev_state = [-1] * S
    prev_op = [-1] * S
    dist[0] = 0
    q = deque([0])
    while q:
        s = q.popleft()
        for c in range(C):
            ns = table[s][c]
            if dist[ns] == -1:
                dist[ns] = dist[s] + 1
                prev_state[ns] = s
                prev_op[ns] = c
                q.append(ns)
    reachable = [s for s in range(S) if s >= 1 and dist[s] != -1]
    if not reachable:
        return []
    target = max(reachable)
    seq = []
    cur = target
    while cur != 0:
        seq.append((cur, prev_op[cur]))
        cur = prev_state[cur]
    seq.reverse()
    return seq


def ancestors_of(t, deps, memo):
    if t in memo:
        return memo[t]
    memo[t] = set()  # cycle guard (schema is a DAG, but be defensive)
    acc = set()
    for d in deps[t]:
        acc.add(d)
        acc |= ancestors_of(d, deps, memo)
    memo[t] = acc
    return acc


def topo_order(nodes, deps):
    nodes = set(nodes)
    indeg = {n: sum(1 for d in deps[n] if d in nodes) for n in nodes}
    children = {n: [] for n in nodes}
    for n in nodes:
        for d in deps[n]:
            if d in nodes:
                children[d].append(n)
    ready = deque(sorted(n for n in nodes if indeg[n] == 0))
    order = []
    while ready:
        n = ready.popleft()
        order.append(n)
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                ready.append(c)
    return order


def main():
    T, C, budget, deps, tables = read_instance()

    value_types = [t for t in range(T) if len(tables[t]) > 1]
    paths = {t: bfs_best_path(tables[t]) for t in value_types}
    value_types = [t for t in value_types if paths[t]]

    memo = {}
    shared_ancestors = set()
    for t in value_types:
        shared_ancestors |= ancestors_of(t, deps, memo)
    shared_ancestors -= set(value_types)

    shared_topo = topo_order(shared_ancestors, deps)

    calls = []
    built = set()
    for a in shared_topo:
        if len(calls) >= budget:
            break
        calls.append(('C', a))
        built.add(a)

    remaining_budget = budget - len(calls)

    items = []  # (type, levels=[(cost,val),...] level index j)
    for t in value_types:
        anc = ancestors_of(t, deps, memo)
        if not anc.issubset(built):
            items.append((t, [(0, 0)]))
            continue
        seq = paths[t]
        levels = [(0, 0)]
        cum = 0
        for j, (state, op) in enumerate(seq, start=1):
            cum += state * state
            levels.append((1 + j, cum))
        items.append((t, levels))

    n = len(items)
    cap = remaining_budget
    NEG = -1
    dp = [[NEG] * (cap + 1) for _ in range(n + 1)]
    dp[0][0] = 0
    for i in range(n):
        _, levels = items[i]
        for c in range(cap + 1):
            if dp[i][c] < 0:
                continue
            base = dp[i][c]
            for (cost_l, val_l) in levels:
                nc = c + cost_l
                if nc <= cap and base + val_l > dp[i + 1][nc]:
                    dp[i + 1][nc] = base + val_l

    best_c = max(range(cap + 1), key=lambda c: dp[n][c])

    chosen_levels = [0] * n
    c = best_c
    for i in range(n - 1, -1, -1):
        _, levels = items[i]
        target = dp[i + 1][c]
        for li in range(len(levels) - 1, -1, -1):
            cost_l, val_l = levels[li]
            if c - cost_l >= 0 and dp[i][c - cost_l] == target - val_l and dp[i][c - cost_l] >= 0:
                chosen_levels[i] = li
                c -= cost_l
                break

    for i in range(n):
        t, levels = items[i]
        lvl = chosen_levels[i]
        if lvl == 0:
            continue
        seq = paths[t][:lvl]
        create_idx = len(calls)
        calls.append(('C', t))
        for (_, op) in seq:
            calls.append(('O', create_idx, op))

    out = [str(len(calls))]
    for call in calls:
        if call[0] == 'C':
            out.append(f"C {call[1]}")
        else:
            out.append(f"O {call[1]} {call[2]}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
