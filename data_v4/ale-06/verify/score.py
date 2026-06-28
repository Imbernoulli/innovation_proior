#!/usr/bin/env python3
"""Deterministic scorer for ale-06 "Production Line Scheduling"
(permutation flow-shop, makespan minimization).

Usage:
    python3 score.py INSTANCE_FILE SOLUTION_FILE [--cmax]

Reads the instance and a candidate solution, validates feasibility, and prints
a single number on stdout:

  * default             -> the ALE-Bench continuous SCORE (higher is better,
                           0.0 if the solution is infeasible).
  * with --cmax flag    -> the raw objective MAKESPAN Cmax (the thing to
                           MINIMIZE). If infeasible, exits non-zero so callers
                           must use the default score for the floor semantics.

Solution format (the solver's stdout):
    a permutation of the n job ids 0..n-1 (whitespace separated, any layout).
    Optionally a leading integer equal to n may be present as a header; the
    scorer accepts EITHER exactly n tokens (the permutation) OR n+1 tokens whose
    first token equals n followed by the permutation. Anything else is
    infeasible.

Feasibility requirements (any violation => SCORE 0.0):
    * the body is exactly a permutation of {0,1,...,n-1}
      (n tokens, each an integer in [0,n-1], all distinct).

Objective (to minimize): the makespan of the permutation flow shop.
Let pi be the chosen job order. With C[i][k] the completion time of the i-th
job of pi on machine k:
    C[0][0]   = p[pi[0]][0]
    C[i][0]   = C[i-1][0] + p[pi[i]][0]                 (first machine)
    C[0][k]   = C[0][k-1] + p[pi[0]][k]                 (first job)
    C[i][k]   = max(C[i-1][k], C[i][k-1]) + p[pi[i]][k]
    Cmax      = C[n-1][m-1]

ALE score (to maximize; this is what the judge reports):
    score = 0.0                                  if infeasible
    score = round(SCORE_SCALE / Cmax)            otherwise
A smaller makespan yields a strictly larger score. Any malformed, wrong-length,
out-of-range, duplicate, or non-permutation output floors the score to 0.0.
"""
import sys

SCORE_SCALE = 1_000_000_000.0  # fixed normalisation constant (frozen)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)
    n = int(next(it))
    m = int(next(it))
    p = []
    for _ in range(n):
        row = [int(next(it)) for _ in range(m)]
        p.append(row)
    return n, m, p


def parse_solution(path, n):
    """Return the permutation as a list of ints, or None if infeasible."""
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if not toks:
        return None
    # Accept an optional leading header equal to n.
    if len(toks) == n + 1:
        try:
            head = int(toks[0])
        except ValueError:
            return None
        if head == n:
            toks = toks[1:]
        else:
            return None
    if len(toks) != n:
        return None
    perm = []
    seen = set()
    for t in toks:
        try:
            v = int(t)
        except ValueError:
            return None
        if v < 0 or v >= n:
            return None
        if v in seen:
            return None  # duplicate -> not a permutation
        seen.add(v)
        perm.append(v)
    if len(seen) != n:
        return None
    return perm


def makespan(n, m, p, perm):
    """Completion time of the last job on the last machine."""
    # C[k] = completion time on machine k of the jobs processed so far.
    C = [0] * m
    for i in range(n):
        job = perm[i]
        prev = 0.0  # completion on the previous machine for this job
        for k in range(m):
            start = C[k] if C[k] > prev else prev
            C[k] = start + p[job][k]
            prev = C[k]
    return C[m - 1]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    want_cmax = "--cmax" in sys.argv[1:]
    if len(args) < 2:
        sys.stderr.write("usage: score.py INSTANCE SOLUTION [--cmax]\n")
        sys.exit(2)
    inst_path, sol_path = args[0], args[1]
    n, m, p = read_instance(inst_path)
    perm = parse_solution(sol_path, n)
    if perm is None:
        if want_cmax:
            sys.stderr.write("infeasible\n")
            sys.exit(1)
        print(0.0)
        return
    cmax = makespan(n, m, p, perm)
    if want_cmax:
        print(int(cmax))
        return
    if cmax <= 0:
        # Degenerate (n==0 or all zero times); avoid division by zero. The
        # generator guarantees n>=40 and times>=1, so this never triggers in
        # practice, but we keep the scorer total.
        print(0.0)
        return
    score = round(SCORE_SCALE / cmax)
    print(score)


if __name__ == "__main__":
    main()
