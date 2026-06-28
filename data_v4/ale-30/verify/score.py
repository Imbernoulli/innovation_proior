#!/usr/bin/env python3
"""
Deterministic local scorer for ale-30: Tower Placement for Signal Coverage.

Usage:
    python3 score.py <instance_file> <solution_file>
prints a single float: the score (0.0 if infeasible).

Scoring rule (as stated in context.md):
  * Read the instance (n, m, req, power matrix P[i][j]).
  * Read the solution: first token is k = number of chosen sites, then k site
    indices (0-based, distinct, each in [0, m)).
  * Accumulate, for each demand i, the received signal
        recv[i] = sum over chosen sites j of P[i][j].
  * FEASIBILITY: the solution is feasible iff
        - k parses, every index is in range and distinct, AND
        - recv[i] >= req[i] - EPS  for EVERY demand i.
    If infeasible in any way (bad parse, out-of-range/duplicate index, or any
    demand unmet), the score is 0.0 (the feasibility -> 0 floor).
  * Otherwise the raw cost is the number of chosen towers, k. We normalise
    against a deterministic GREEDY MAX-COVERAGE baseline computed here (the
    classic set-cover greedy: repeatedly add the site that reduces the total
    remaining deficit the most, until all demands are met). The score is

        score = baseline_towers / k

    Fewer towers => higher score; the greedy baseline scores 1.0 by definition,
    and an optimal/LP-rounded solver scores > 1.0.
"""
import sys

EPS = 1e-6


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    req = [float(next(it)) for _ in range(n)]
    P = [[float(next(it)) for _ in range(m)] for _ in range(n)]
    return n, m, req, P


def greedy_baseline(n, m, req, P):
    """Classic deterministic set-cover greedy on the remaining-deficit metric.

    deficit[i] = max(0, req[i] - recv[i]); pick the site that reduces
    sum(deficit) the most; ties broken by smallest site index. Returns the
    number of towers the greedy uses (guaranteed to terminate because the
    all-sites set is feasible by construction).
    """
    recv = [0.0] * n
    chosen = [False] * m
    count = 0
    # remaining deficit
    def total_deficit():
        return sum(max(0.0, req[i] - recv[i]) for i in range(n))

    while total_deficit() > EPS:
        best_j = -1
        best_gain = -1.0
        for j in range(m):
            if chosen[j]:
                continue
            gain = 0.0
            for i in range(n):
                d = req[i] - recv[i]
                if d > 0.0:
                    gain += min(d, P[i][j])
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_j = j
        if best_j < 0 or best_gain <= 1e-12:
            # No site can make further progress; should not happen (all-on
            # is feasible), but guard against an infinite loop.
            break
        chosen[best_j] = True
        count += 1
        for i in range(n):
            recv[i] += P[i][best_j]
    return count


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    inst, soln = sys.argv[1], sys.argv[2]
    n, m, req, P = read_instance(inst)

    # Parse solution.
    try:
        with open(soln) as f:
            toks = f.read().split()
        if not toks:
            print(0.0)
            return
        it = iter(toks)
        k = int(next(it))
        if k < 0:
            print(0.0)
            return
        idx = [int(next(it)) for _ in range(k)]
    except Exception:
        print(0.0)
        return

    # Validate indices: in range and distinct.
    seen = set()
    for j in idx:
        if j < 0 or j >= m or j in seen:
            print(0.0)
            return
        seen.add(j)

    # Accumulate received signal.
    recv = [0.0] * n
    for j in idx:
        col = P  # P[i][j]
        for i in range(n):
            recv[i] += P[i][j]

    # Feasibility check.
    for i in range(n):
        if recv[i] < req[i] - EPS:
            print(0.0)
            return

    if k == 0:
        # Feasible with zero towers only if all req are <= 0; then any positive
        # baseline beats it -> treat as a perfect (cost 1 floor) to avoid div0.
        k_eff = 1
    else:
        k_eff = k

    baseline = greedy_baseline(n, m, req, P)
    if baseline <= 0:
        baseline = 1
    score = baseline / float(k_eff)
    print(f"{score:.6f}")


if __name__ == "__main__":
    main()
