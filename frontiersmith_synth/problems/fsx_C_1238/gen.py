#!/usr/bin/env python3
"""gen.py <testId> -> prints ONE feature-flag rollout instance to stdout.

Deterministic: instance depends ONLY on testId (seeded RNG).
"""
import sys
import random

# difficulty ladder: number of flags per testId (1..10, small->large/adversarial)
SIZES = {1: 4, 2: 5, 3: 6, 4: 6, 5: 8, 6: 8, 7: 9, 8: 10, 9: 11, 10: 12}

# For each testId, the sizes of disjoint fully-conflicting cliques planted
# among the highest-value flags. testId 1 is a pure independent warm-up
# (greedy is optimal there); every later case plants at least one clique so
# the value-greedy rollback trap fires on the clear majority of cases.
CLIQUE_PLAN = {
    1: [],
    2: [2],
    3: [3],
    4: [2, 2],
    5: [4],
    6: [4, 2],
    7: [5, 2],
    8: [4, 3],
    9: [6, 2],
    10: [6, 3],
}


def build(test_id):
    rnd = random.Random(20260726 + test_id * 104729 + 17)
    N = SIZES[test_id]
    # protected (baseline-defining) flags carry small, modest values; every
    # other flag -- including all clique members -- is drawn from a much
    # higher range, so the value a naive greedy chases (and loses to
    # rollback thrash) dwarfs the checker's weak internal baseline.
    values = [0] * N
    for i in range(1, N + 1):
        if i in (1, 2):
            values[i - 1] = rnd.randint(5, 15)
        else:
            values[i - 1] = rnd.randint(30, 100)

    # requires forest: parent index always < child index (guarantees acyclic
    # and that plain index order is a valid topological order)
    parent = [0] * (N + 1)
    protected = {1, 2}  # flags 1,2 are always untouched roots -> baseline B > 0
    req_prob = 0.35 if test_id <= 4 else 0.45
    for i in range(3, N + 1):
        if rnd.random() < req_prob:
            parent[i] = rnd.randint(1, i - 1)

    conflicts = set()

    def add_conflict(a, b):
        if a == b:
            return
        if a in protected or b in protected:
            return
        if parent[a] == b or parent[b] == a:
            return  # never contradict a requires edge with a conflict edge
        conflicts.add((min(a, b), max(a, b)))

    # plant disjoint fully-conflicting cliques among the highest-value flags:
    # each clique forces "at most one of these ever active", which the naive
    # value-order-plus-rollback greedy resolves by cycling through every
    # member from highest to lowest value, burning a rollback window per
    # swap and ending up stuck with the LOWEST-value member active.
    pool = sorted(range(3, N + 1), key=lambda i: -values[i - 1])
    p = 0
    for size in CLIQUE_PLAN[test_id]:
        clique = pool[p:p + size]
        p += size
        for a in range(len(clique)):
            for b in range(a + 1, len(clique)):
                add_conflict(clique[a], clique[b])

    requires = [(i, parent[i]) for i in range(1, N + 1) if parent[i] != 0]
    conflicts_list = sorted(conflicts)
    return N, values, requires, conflicts_list


def main():
    test_id = int(sys.argv[1])
    N, values, requires, conflicts = build(test_id)
    out = []
    out.append(str(N))
    out.append(' '.join(str(v) for v in values))
    out.append(str(len(requires)))
    for c, p in requires:
        out.append(f"{c} {p}")
    out.append(str(len(conflicts)))
    for a, b in conflicts:
        out.append(f"{a} {b}")
    sys.stdout.write('\n'.join(out) + '\n')


if __name__ == '__main__':
    main()
