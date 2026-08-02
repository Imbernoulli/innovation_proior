#!/usr/bin/env python3
"""
verify.py <in> <out> <ans>  (ans ignored)

Deterministic scorer for the chain-join adaptive-plan problem.

Instance (<in>):
    testId
    n
    C_0 .. C_{n-1}                  exact base cardinalities
    MEM_CAP  SPILL_MULT
    h                                reoptimization checkpoint depth (2<=h<=n-1)
    (n-1 lines) S_est[e]  F[e]       e = 0..n-2, edge between relation e,e+1

Artifact (<out>), whitespace-tokenized:
    START <s>
    PRE <h-1 tokens of L/R>
    BRANCH LOW  <n-h tokens of L/R>
    BRANCH MID  <n-h tokens of L/R>
    BRANCH HIGH <n-h tokens of L/R>
(the three BRANCH blocks may appear in any order, each bucket exactly once)

'L' extends the current joined interval one relation to the left, 'R' one
relation to the right, starting from the singleton interval {s}.

Cost model: each extension step's cost is its resulting intermediate row
count, times SPILL_MULT if that count exceeds MEM_CAP (disk spill).

The hidden TRUE selectivity of edge e (NEVER printed to stdin) is realized
deterministically from (testId, e, S_est[e], F[e]) below: an edge whose
stated bound is huge (F>=10) always realizes at the EXPENSIVE extreme
(min(1, S_est*F)); a mildly-uncertain edge realizes a seeded-random point
inside its certified bound. Feasibility, after simulating the PREFIX then
the BRANCH matching the true prefix bucket:
  - every move stays in [0, n-1];
  - the plan covers all n relations exactly once (prefix + chosen branch).
Objective (minimize) F = total true row-touch cost of prefix + chosen branch.
Baseline B = true cost of the canonical non-adaptive plan (start at 0,
always extend right). sc = min(900, 100*B/F); Ratio = sc/1000.
"""
import sys, random


def fail(msg):
    print("Ratio: 0.0  (%s)" % msg)
    sys.exit(0)


def true_selectivity(tid, e, s_est, f):
    if f >= 10.0:
        return min(1.0, s_est * f)
    rng = random.Random(500009 + tid * 104729 + e * 7907)
    mult = rng.uniform(1.0 / f, f)
    return min(1.0, max(1e-6, s_est * mult))


def run_from(n, C, lo, hi, size, moves, sel, mem_cap, spill_mult):
    cost = 0.0
    for mv in moves:
        if mv == 'L':
            if lo == 0:
                return None
            edge = lo - 1
            lo -= 1
            size = size * C[lo] * sel(edge)
        elif mv == 'R':
            if hi == n - 1:
                return None
            edge = hi
            hi += 1
            size = size * C[hi] * sel(edge)
        else:
            return None
        step_cost = size if size <= mem_cap else size * spill_mult
        cost += step_cost
    return lo, hi, size, cost


def main():
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        itok = open(inf).read().split()
    except Exception:
        fail("cannot read input")
    if not itok:
        fail("empty input")
    idx = 0
    tid = int(itok[idx]); idx += 1
    n = int(itok[idx]); idx += 1
    if n < 3 or n > 20:
        fail("bad n")
    C = [int(itok[idx + i]) for i in range(n)]; idx += n
    mem_cap = float(itok[idx]); idx += 1
    spill_mult = float(itok[idx]); idx += 1
    h = int(itok[idx]); idx += 1
    if not (2 <= h <= n - 1):
        fail("bad checkpoint depth")
    est, bound = [], []
    for e in range(n - 1):
        s_est = float(itok[idx]); idx += 1
        f = float(itok[idx]); idx += 1
        est.append(s_est); bound.append(f)

    def true_sel(e):
        return true_selectivity(tid, e, est[e], bound[e])

    try:
        otok = open(outf).read().split()
    except Exception:
        fail("cannot read output")
    if len(otok) < 4:
        fail("output too short")

    oi = 0
    def nxt():
        nonlocal oi
        if oi >= len(otok):
            return None
        t = otok[oi]; oi += 1
        return t

    if nxt() != "START":
        fail("expected START")
    stok = nxt()
    if stok is None:
        fail("missing start relation")
    try:
        s = int(stok)
    except ValueError:
        fail("start relation is not an integer")
    if not (0 <= s < n):
        fail("start relation out of range")

    if nxt() != "PRE":
        fail("expected PRE")
    pre_moves = []
    for _ in range(h - 1):
        t = nxt()
        if t not in ('L', 'R'):
            fail("PRE move must be L/R")
        pre_moves.append(t)

    need = n - h
    branches = {}
    for _ in range(3):
        if nxt() != "BRANCH":
            fail("expected BRANCH")
        name = nxt()
        if name not in ("LOW", "MID", "HIGH"):
            fail("unknown branch name %r" % name)
        if name in branches:
            fail("duplicate branch %s" % name)
        moves = []
        for _ in range(need):
            t = nxt()
            if t not in ('L', 'R'):
                fail("BRANCH move must be L/R")
            moves.append(t)
        branches[name] = moves

    if oi != len(otok):
        fail("trailing tokens in output")
    if set(branches.keys()) != {"LOW", "MID", "HIGH"}:
        fail("must provide exactly the LOW, MID and HIGH branches")

    pre_res = run_from(n, C, s, s, float(C[s]), pre_moves, true_sel, mem_cap, spill_mult)
    if pre_res is None:
        fail("PRE plan runs out of chain bounds")
    lo, hi, size, pre_cost = pre_res
    if hi - lo + 1 != h:
        fail("PRE plan does not cover exactly h relations")

    if size <= mem_cap:
        bucket = "LOW"
    elif size <= 4.0 * mem_cap:
        bucket = "MID"
    else:
        bucket = "HIGH"

    cont_res = run_from(n, C, lo, hi, size, branches[bucket], true_sel, mem_cap, spill_mult)
    if cont_res is None:
        fail("chosen BRANCH (%s) runs out of chain bounds" % bucket)
    flo, fhi, fsize, cont_cost = cont_res
    if not (flo == 0 and fhi == n - 1):
        fail("final plan does not cover all n relations")

    F = pre_cost + cont_cost
    if F != F or F in (float("inf"), float("-inf")) or F <= 0:
        fail("non-finite or non-positive objective")

    base_res = run_from(n, C, 0, 0, float(C[0]), ['R'] * (n - 1), true_sel, mem_cap, spill_mult)
    B = base_res[3]

    sc = min(900.0, 100.0 * B / max(1e-9, F))
    print("bucket=%s F=%.3f B=%.3f  Ratio: %.6f" % (bucket, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
