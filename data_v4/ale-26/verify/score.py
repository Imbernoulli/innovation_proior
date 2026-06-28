#!/usr/bin/env python3
"""Deterministic local scorer for "Soft-Constraint Assignment".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single integer: the score. A higher score is better.

Problem recap (see context.md "Evaluation settings"):
  * n agents, m slots; slot j has capacity cap_j (sum cap_j >= n is guaranteed).
  * pref[i][j] is the (integer) preference of putting agent i in slot j.
  * C soft constraints, each `t a b w`:
        t == 0 (DIFFER): penalty w iff assign[a] == assign[b];
        t == 1 (SAME)  : penalty w iff assign[a] != assign[b].
  * A SOLUTION is n integers s_0 ... s_{n-1}, where s_i in [0, m) is the slot of
    agent i.
  * OBJECTIVE of a feasible solution:
        P = sum_i pref[i][s_i]  -  sum over violated constraints of their w.
    P may be positive or (in principle) negative -- it is NOT the score directly.

FEASIBILITY (floor to 0):
  The output must be EXACTLY n integer tokens, each in [0, m), AND for every slot j
  the number of agents assigned to j must be <= cap_j (no capacity overflow). If any
  of this fails -- wrong token count, an out-of-range slot, a non-integer token, a
  capacity overflow, a missing file -- the solution is INFEASIBLE and the score is 0.

SCORE (deterministic, reproducible, scale-invariant):
  Let
    * P      = the submitted solution's objective (above),
    * P_base = the objective of the scorer's own deterministic GREEDY-BEST-PREFERENCE
               baseline (recomputed here, independent of the solver): assign every
               agent to its highest-preference slot ignoring constraints, then repair
               any capacity overflow by evicting, from each over-full slot, the agents
               that lose the LEAST preference by moving to their best still-feasible
               slot. This baseline ignores soft constraints entirely, so a real solver
               must EARN its lead by trading a little preference for fewer violations.
    * Scale  = sum_i max_j pref[i][j]  (the maximum attainable raw preference; a
               positive instance-scale normalizer, > 0 for n >= 1 since prefs are
               non-negative and each agent has at least one slot with pref >= 0; we
               also floor it at 1 to avoid division by zero on all-zero prefs).
  Then
        score = round( 1_000_000 + 1_000_000 * (P - P_base) / Scale ),  clamped >= 0.
  The greedy-best-preference baseline scores exactly 1_000_000. A constraint-aware
  solver that improves the objective scores strictly more; a worse one scores less but
  never below 0. INFEASIBLE -> 0.

The scorer does NOT trust the solver: it recomputes P_base and Scale itself, and it
recomputes every penalty from the constraint list.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    cap = [int(next(it)) for _ in range(m)]
    pref = [[int(next(it)) for _ in range(m)] for _ in range(n)]
    C = int(next(it))
    cons = []
    for _ in range(C):
        t = int(next(it))
        a = int(next(it))
        b = int(next(it))
        w = int(next(it))
        cons.append((t, a, b, w))
    return n, m, cap, pref, cons


def read_solution(path, n, m, cap):
    """Return assign[0..n-1] (each in [0,m), capacities respected) if well-formed,
    else None (-> infeasible -> score 0)."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if len(toks) != n:
        return None  # must be exactly n tokens
    assign = [0] * n
    load = [0] * m
    for i, tk in enumerate(toks):
        try:
            v = int(tk)
        except ValueError:
            return None
        if v < 0 or v >= m:
            return None
        assign[i] = v
        load[v] += 1
        if load[v] > cap[v]:
            return None  # capacity overflow -> infeasible
    return assign


def objective(assign, pref, cons):
    """P = sum chosen prefs - sum violated penalties."""
    P = 0
    for i, s in enumerate(assign):
        P += pref[i][s]
    for (t, a, b, w) in cons:
        if t == 0:           # DIFFER: penalty if same
            if assign[a] == assign[b]:
                P -= w
        else:                # SAME: penalty if different
            if assign[a] != assign[b]:
                P -= w
    return P


def greedy_best_preference(n, m, cap, pref):
    """Deterministic baseline: each agent to its best slot (ignoring constraints),
    then repair capacity overflow by evicting the least-loss agents to their best
    still-feasible slot. Returns the assignment (always capacity-feasible because
    sum(cap) >= n is guaranteed)."""
    # best slot per agent (ties -> smallest index)
    assign = [0] * n
    load = [0] * m
    for i in range(n):
        best_j = 0
        best_v = pref[i][0]
        for j in range(1, m):
            if pref[i][j] > best_v:
                best_v = pref[i][j]
                best_j = j
        assign[i] = best_j
        load[best_j] += 1

    # repair overflow deterministically
    # Process slots in index order; while a slot is over capacity, evict the agent
    # whose "loss" (pref at current slot minus best pref at a still-feasible slot) is
    # smallest, breaking ties by agent index.
    remaining_cap = [cap[j] - load[j] for j in range(m)]  # may be negative on over-full

    for j in range(m):
        while load[j] > cap[j]:
            # agents currently in slot j
            members = [i for i in range(n) if assign[i] == j]
            best_choice = None  # (loss, agent, target)
            for i in members:
                cur = pref[i][j]
                # best alternative slot with spare capacity (remaining_cap > 0), != j
                tj = -1
                tv = None
                for k in range(m):
                    if k == j:
                        continue
                    if remaining_cap[k] > 0:
                        if tv is None or pref[i][k] > tv or (pref[i][k] == tv and (tj == -1 or k < tj)):
                            tv = pref[i][k]
                            tj = k
                if tj == -1:
                    continue  # nowhere to move this agent right now
                loss = cur - tv
                cand = (loss, i, tj)
                if best_choice is None or cand < best_choice:
                    best_choice = cand
            if best_choice is None:
                # No spare capacity anywhere reachable -- impossible since
                # sum(cap) >= n, but guard against an infinite loop.
                break
            _, i, tj = best_choice
            assign[i] = tj
            load[j] -= 1
            load[tj] += 1
            remaining_cap[tj] -= 1
            remaining_cap[j] += 1
    return assign


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    n, m, cap, pref, cons = read_instance(sys.argv[1])

    assign = read_solution(sys.argv[2], n, m, cap)
    if assign is None:
        print(0)  # INFEASIBLE -> floored to 0
        return

    P = objective(assign, pref, cons)

    base_assign = greedy_best_preference(n, m, cap, pref)
    P_base = objective(base_assign, pref, cons)

    Scale = sum(max(pref[i]) for i in range(n))
    if Scale < 1:
        Scale = 1

    score = 1_000_000.0 + 1_000_000.0 * (P - P_base) / Scale
    if score < 0.0:
        score = 0.0
    print(int(round(score)))


if __name__ == "__main__":
    main()
