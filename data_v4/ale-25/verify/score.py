#!/usr/bin/env python3
"""Deterministic local scorer for "Interval Scheduling on Few Rooms" (ale-25).

Usage:
    python3 score.py INSTANCE_FILE SOLUTION_FILE [--raw]

Scoring rule (matches context.md):
  * The instance has n intervals and K rooms.
  * The solution is n integers r_0..r_{n-1}; r_i in {-1, 0, 1, ..., K-1}.
    r_i = -1 means interval i is rejected; otherwise interval i is assigned to
    room r_i (and thereby accepted).
  * FEASIBILITY: within any single room, no two assigned intervals may overlap.
    Intervals are treated as half-open [start, end): two intervals overlap iff
    a.start < b.end and b.start < a.end. Sharing only an endpoint (a.end ==
    b.start) is allowed.
  * OBJECTIVE: the sum of weights of all accepted intervals.
  * FLOOR: if the solution is malformed (wrong count, out-of-range room id) OR
    any room contains two overlapping accepted intervals, the score is 0.

By default this prints the NORMALIZED score: raw_objective / first_fit_baseline,
where the baseline is the deterministic "first-fit by start time" assignment
(sort accepted-eligible intervals by start, place each in the lowest-indexed
room whose last end <= its start, else reject). With --raw it prints the raw
objective instead. An infeasible solution always prints 0.0 (or 0).
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    K = int(toks[idx]); idx += 1
    intervals = []
    for _ in range(n):
        s = int(toks[idx]); e = int(toks[idx + 1]); w = int(toks[idx + 2])
        idx += 3
        intervals.append((s, e, w))
    return n, K, intervals


def read_solution(path, n):
    with open(path) as f:
        toks = f.read().split()
    return [int(t) for t in toks], n


def raw_objective(n, K, intervals, assign):
    """Return raw accepted weight, or None if infeasible/malformed."""
    if len(assign) != n:
        return None
    rooms = [[] for _ in range(K)]
    total = 0
    for i, r in enumerate(assign):
        if r == -1:
            continue
        if r < 0 or r >= K:
            return None
        s, e, w = intervals[i]
        rooms[r].append((s, e))
        total += w
    # feasibility: per-room, sort by start, check no overlap (half-open)
    for r in range(K):
        ivs = sorted(rooms[r])
        for j in range(1, len(ivs)):
            prev_e = ivs[j - 1][1]
            cur_s = ivs[j][0]
            if cur_s < prev_e:      # overlap (touching endpoints allowed)
                return None
    return total


def first_fit_baseline(n, K, intervals):
    """Deterministic first-fit-by-start assignment; returns its raw objective."""
    order = sorted(range(n), key=lambda i: (intervals[i][0], intervals[i][1]))
    last_end = [0] * K  # rooms start free; use -inf via -1 sentinel
    last_end = [-1] * K
    total = 0
    for i in order:
        s, e, w = intervals[i]
        placed = False
        for r in range(K):
            if last_end[r] <= s:
                last_end[r] = e
                total += w
                placed = True
                break
        # else rejected
    return total


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py INSTANCE SOLUTION [--raw]\n")
        sys.exit(2)
    inst_path, sol_path = sys.argv[1], sys.argv[2]
    raw_mode = "--raw" in sys.argv[3:]

    n, K, intervals = read_instance(inst_path)
    assign, _ = read_solution(sol_path, n)

    obj = raw_objective(n, K, intervals, assign)
    if obj is None:
        print(0 if raw_mode else 0.0)
        return

    if raw_mode:
        print(obj)
        return

    base = first_fit_baseline(n, K, intervals)
    if base <= 0:
        # degenerate instance; report raw so we never divide by zero
        print(float(obj))
        return
    print(obj / base)


if __name__ == "__main__":
    main()
