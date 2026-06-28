#!/usr/bin/env python3
"""Deterministic local scorer for "Knapsack with Synergies" (Quadratic Knapsack).

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single number: the score (an integer). A higher score is better.

Scoring rule (see context.md "Evaluation settings"):
  * The instance is a Quadratic Knapsack Problem (QKP): n items, each with a
    positive weight w_i and a non-negative linear value v_i, a weight budget W,
    and p synergy pairs {i,j} each with a positive bonus b that is earned iff
    BOTH i and j are selected.
  * A SOLUTION is a chosen subset S of items, given as a count k followed by k
    distinct item indices. Its OBJECTIVE is
        obj(S) = sum_{i in S} v_i  +  sum_{ {i,j} : i in S and j in S } b_{ij}.
  * FEASIBILITY: S is feasible iff (a) the output parses as k followed by exactly
    k integer indices, (b) every index is in [0, n) and the k indices are
    DISTINCT, and (c) the total selected weight sum_{i in S} w_i <= W.
    The empty set (k = 0) is feasible and has objective 0.
  * FEASIBILITY FLOOR: if the output does not parse, repeats an index, names an
    out-of-range index, or exceeds the weight budget W, the solution is
    INFEASIBLE and the score is 0.
  * REFERENCE BASELINE G (recomputed by the scorer, solver-independent): the
    objective of the classic value/weight-RATIO greedy that IGNORES synergies --
    sort items by v_i / w_i descending (ties by larger v_i, then smaller index)
    and add each item whose weight still fits, then add the synergy bonuses that
    happen to be realized among the chosen items. The scorer computes this
    itself so the reference is reproducible and does not trust the solver.
  * SCORE:
        score = round(1_000_000 * obj(S) / G)   for a feasible S with G > 0
        score = round(1_000_000 * obj(S))       for a feasible S if G == 0 (rare)
        score = 0                                if INFEASIBLE
    The ratio-greedy reference scores exactly 1_000_000; a synergy-aware subset
    that collects more bonus scores strictly more; a weaker feasible subset
    scores less but stays positive. Infeasible -> 0.

The scorer is self-contained and deterministic: it does not trust the solver and
recomputes G itself.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    W = int(next(it))
    w = [0] * n
    v = [0] * n
    for i in range(n):
        w[i] = int(next(it))
        v[i] = int(next(it))
    p = int(next(it))
    pairs = []
    for _ in range(p):
        a = int(next(it))
        b = int(next(it))
        bonus = int(next(it))
        pairs.append((a, b, bonus))
    return n, W, w, v, pairs


def read_solution(path):
    """Return a list of int indices, or None on a parse error."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if not toks:
        return None
    it = iter(toks)
    try:
        k = int(next(it))
    except (StopIteration, ValueError):
        return None
    if k < 0:
        return None
    idxs = []
    for _ in range(k):
        try:
            idxs.append(int(next(it)))
        except (StopIteration, ValueError):
            return None
    # Any trailing tokens beyond the declared k indices are an error.
    if next(it, None) is not None:
        return None
    return idxs


def objective(n, w, v, pairs, idxs):
    """Return (feasible, weight, obj). Does NOT check the budget here."""
    sel = [False] * n
    for i in idxs:
        if i < 0 or i >= n:
            return False, 0, 0          # out-of-range index
        if sel[i]:
            return False, 0, 0          # repeated index
        sel[i] = True
    weight = sum(w[i] for i in idxs)
    obj = sum(v[i] for i in idxs)
    for (a, b, bonus) in pairs:
        if sel[a] and sel[b]:
            obj += bonus
    return True, weight, obj


def ratio_greedy_objective(n, W, w, v, pairs):
    """Value/weight-ratio greedy that IGNORES synergies, then count realized
    synergy bonuses among the items it happened to pick. Reproducible reference.

    Deterministic ordering: by v_i / w_i descending, tie-break by larger v_i,
    then by smaller index. Compared with cross-multiplication to avoid any float
    nondeterminism.
    """
    order = list(range(n))
    # key: ratio desc -> -(v_i / w_i); we sort via a comparator-free key using
    # exact fractions through cross multiplication by sorting on a tuple.
    # Python's sort is stable; we build a key whose primary component is the
    # ratio as a Fraction-free float is risky, so use a custom sort.
    import functools

    def cmp(i, j):
        # ratio_i = v_i / w_i  vs  ratio_j = v_j / w_j ; both w > 0
        lhs = v[i] * w[j]
        rhs = v[j] * w[i]
        if lhs != rhs:
            return -1 if lhs > rhs else 1     # larger ratio first
        if v[i] != v[j]:
            return -1 if v[i] > v[j] else 1   # larger value first
        return -1 if i < j else (1 if i > j else 0)

    order.sort(key=functools.cmp_to_key(cmp))

    sel = [False] * n
    weight = 0
    for i in order:
        if weight + w[i] <= W:
            sel[i] = True
            weight += w[i]
    obj = sum(v[i] for i in range(n) if sel[i])
    for (a, b, bonus) in pairs:
        if sel[a] and sel[b]:
            obj += bonus
    return obj


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, W, w, v, pairs = read_instance(sys.argv[1])

    idxs = read_solution(sys.argv[2])
    if idxs is None:
        print(0)  # INFEASIBLE -> floored to 0
        return

    feasible, weight, obj = objective(n, w, v, pairs, idxs)
    if not feasible:
        print(0)
        return
    if weight > W:
        print(0)  # over budget -> infeasible
        return

    G = ratio_greedy_objective(n, W, w, v, pairs)
    if G > 0:
        score = int(round(1_000_000.0 * obj / G))
    else:
        # Degenerate: greedy collects nothing; fall back to absolute objective.
        score = int(round(1_000_000.0 * obj))
    print(score)


if __name__ == "__main__":
    main()
