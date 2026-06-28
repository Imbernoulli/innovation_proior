#!/usr/bin/env python3
"""Deterministic local scorer for "Time-Indexed Crew Rostering".

Usage:
    python3 score.py <instance_file> <solution_file>

Prints a single number: the score (an integer). Higher is better.

THE MODEL
  We assign each (worker w, day d) a shift code a[w][d] in {0, 1, ..., S}, where 0
  means "day off" and s in 1..S means worker w works shift s on day d. A SOLUTION
  is the W x D grid of these codes (W rows, D integers each).

  Each shift s has a length HOURS[s] and a clock start START[s]; its clock end is
  END[s] = START[s] + HOURS[s]. The rest gap between a shift s worked on day d and a
  shift s' worked on day d+1 is REST(s, s') = 24 + START[s'] - END[s] hours.

FEASIBILITY (any violation => score 0, the feasibility floor)
  The grid must parse as exactly W rows of D integers, each in [0, S]. In addition,
  for every worker w the assignment must obey ALL hard rest rules:
    (A) Availability: if a[w][d] = s >= 1 then AVAIL[w][d][s] must be 1.
    (B) Min rest:     for consecutive working days, if a[w][d] = s >= 1 and
                      a[w][d+1] = s' >= 1 then REST(s, s') >= MIN_REST.
    (C) Max consecutive working days: w may not work more than MAXCONS days in a row.
    (D) Hard weekly hours: the worker's total assigned hours must be <= HARDH.
  If any of (A)-(D) fails for any worker, OR the grid is malformed (wrong shape,
  token not an integer, code out of [0, S], missing file), the score is 0.

OBJECTIVE (for a feasible grid)
  Coverage value: for each (day d, shift s) let cov = number of workers assigned
  shift s on day d. The covered units are min(cov, DEMAND[d][s]) (covering beyond
  demand is wasted overstaffing and earns nothing). Coverage value is
      COV = sum over d, s of VALUE[d][s] * min(cov[d][s], DEMAND[d][s]).
  Overtime penalty: for each worker, hours beyond the soft cap MAXH (but within the
  hard cap HARDH) are overtime; total overtime hours OT are penalised at LAMBDA each.
      OBJ = COV - LAMBDA * OT.

NORMALISATION (reported score)
  The scorer recomputes its own deterministic GREEDY FILL-BY-DEMAND roster G (see
  greedy_roster below), which is always feasible, and reports
      score = round(1_000_000 * OBJ / OBJ(G))           if OBJ(G) > 0
      score = round(1_000_000 * OBJ) ... (degenerate)    if OBJ(G) <= 0 (clamped >= 0)
  Infeasible solutions score 0. The greedy reference scores ~1_000_000; a roster
  that covers more high-value shifts with less overtime scores strictly higher.
  The scorer does not trust the solver: it recomputes the greedy reference itself.
"""
import sys


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    it = iter(toks)

    def nxt():
        return int(next(it))

    W = nxt(); D = nxt(); S = nxt()
    HOURS = [0] + [nxt() for _ in range(S)]      # 1-indexed; HOURS[0] unused
    START = [0] + [nxt() for _ in range(S)]
    MIN_REST = nxt(); MAXCONS = nxt(); MAXH = nxt(); HARDH = nxt(); LAMBDA = nxt()
    demand = [[0] * (S + 1) for _ in range(D)]
    value = [[0] * (S + 1) for _ in range(D)]
    for d in range(D):
        for s in range(1, S + 1):
            demand[d][s] = nxt()
        for s in range(1, S + 1):
            value[d][s] = nxt()
    avail = [[[0] * (S + 1) for _ in range(D)] for _ in range(W)]
    for w in range(W):
        for d in range(D):
            for s in range(1, S + 1):
                avail[w][d][s] = nxt()
    inst = dict(W=W, D=D, S=S, HOURS=HOURS, START=START, MIN_REST=MIN_REST,
                MAXCONS=MAXCONS, MAXH=MAXH, HARDH=HARDH, LAMBDA=LAMBDA,
                demand=demand, value=value, avail=avail)
    return inst


def end_hour(inst, s):
    return inst["START"][s] + inst["HOURS"][s]


def rest_gap(inst, s, s2):
    # gap between shift s on day d and shift s2 on day d+1
    return 24 + inst["START"][s2] - end_hour(inst, s)


def read_solution(path, inst):
    """Return a W x D grid of ints in [0,S] if well-formed, else None."""
    W, D, S = inst["W"], inst["D"], inst["S"]
    try:
        with open(path) as f:
            toks = f.read().split()
    except OSError:
        return None
    if len(toks) != W * D:
        return None
    grid = [[0] * D for _ in range(W)]
    k = 0
    for w in range(W):
        for d in range(D):
            try:
                v = int(toks[k])
            except ValueError:
                return None
            if v < 0 or v > S:
                return None
            grid[w][d] = v
            k += 1
    return grid


def is_feasible(inst, grid):
    W, D, S = inst["W"], inst["D"], inst["S"]
    MIN_REST, MAXCONS, HARDH = inst["MIN_REST"], inst["MAXCONS"], inst["HARDH"]
    HOURS, avail = inst["HOURS"], inst["avail"]
    for w in range(W):
        row = grid[w]
        consec = 0
        total_h = 0
        for d in range(D):
            s = row[d]
            if s == 0:
                consec = 0
                continue
            # (A) availability
            if avail[w][d][s] != 1:
                return False
            total_h += HOURS[s]
            consec += 1
            # (C) max consecutive working days
            if consec > MAXCONS:
                return False
            # (B) min rest vs the next day, if that day is also worked
            if d + 1 < D:
                s2 = row[d + 1]
                if s2 != 0 and rest_gap(inst, s, s2) < MIN_REST:
                    return False
        # (D) hard weekly hours
        if total_h > HARDH:
            return False
    return True


def objective(inst, grid):
    """Coverage value minus overtime penalty. Assumes grid is feasible."""
    W, D, S = inst["W"], inst["D"], inst["S"]
    HOURS, MAXH, LAMBDA = inst["HOURS"], inst["MAXH"], inst["LAMBDA"]
    demand, value = inst["demand"], inst["value"]
    cov = [[0] * (S + 1) for _ in range(D)]
    overtime = 0
    for w in range(W):
        total_h = 0
        for d in range(D):
            s = grid[w][d]
            if s >= 1:
                cov[d][s] += 1
                total_h += HOURS[s]
        if total_h > MAXH:
            overtime += total_h - MAXH
    cover_value = 0
    for d in range(D):
        for s in range(1, S + 1):
            cover_value += value[d][s] * min(cov[d][s], demand[d][s])
    return cover_value - LAMBDA * overtime


def greedy_roster(inst):
    """Deterministic GREEDY FILL-BY-DEMAND reference roster (always feasible).

    Process (day, shift) slots in a fixed order of decreasing value (ties broken by
    (day, shift)). For each slot, fill it up to its demand by assigning the lowest-
    indexed available workers who can take it WITHOUT breaking any hard rule (A)-(D),
    given the assignments already made. This mirrors the natural "staff the most
    valuable shifts first" baseline and is reproducible inside the scorer.
    """
    W, D, S = inst["W"], inst["D"], inst["S"]
    HOURS, HARDH = inst["HOURS"], inst["HARDH"]
    MAXCONS, MIN_REST = inst["MAXCONS"], inst["MIN_REST"]
    demand, value, avail = inst["demand"], inst["value"], inst["avail"]
    grid = [[0] * D for _ in range(W)]
    hours_used = [0] * W

    def ok_to_assign(w, d, s):
        if avail[w][d][s] != 1:
            return False
        if grid[w][d] != 0:
            return False
        if hours_used[w] + HOURS[s] > HARDH:
            return False
        # rest vs previous and next day
        if d - 1 >= 0:
            sp = grid[w][d - 1]
            if sp != 0 and rest_gap(inst, sp, s) < MIN_REST:
                return False
        if d + 1 < D:
            sn = grid[w][d + 1]
            if sn != 0 and rest_gap(inst, s, sn) < MIN_REST:
                return False
        # max consecutive days: count run length if we place s at d
        left = 0
        dd = d - 1
        while dd >= 0 and grid[w][dd] != 0:
            left += 1
            dd -= 1
        right = 0
        dd = d + 1
        while dd < D and grid[w][dd] != 0:
            right += 1
            dd += 1
        if left + 1 + right > MAXCONS:
            return False
        return True

    slots = [(d, s) for d in range(D) for s in range(1, S + 1)]
    slots.sort(key=lambda ds: (-value[ds[0]][ds[1]], ds[0], ds[1]))
    for (d, s) in slots:
        need = demand[d][s]
        filled = 0
        for w in range(W):
            if filled >= need:
                break
            if ok_to_assign(w, d, s):
                grid[w][d] = s
                hours_used[w] += HOURS[s]
                filled += 1
    return grid


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("usage: score.py <instance> <solution>\n")
        sys.exit(1)
    inst = read_instance(sys.argv[1])

    grid = read_solution(sys.argv[2], inst)
    if grid is None or not is_feasible(inst, grid):
        print(0)  # INFEASIBLE -> floored to 0
        return

    obj = objective(inst, grid)

    g = greedy_roster(inst)
    g_obj = objective(inst, g)

    if g_obj > 0:
        score = int(round(1_000_000.0 * obj / g_obj))
    else:
        score = int(round(1_000_000.0 * obj))
    if score < 0:
        score = 0
    print(score)


if __name__ == "__main__":
    main()
