# TIER: strong
"""Robust + adaptive query-plan selection.

Insight: choosing the plan that is cheapest under the POINT-ESTIMATE
selectivity is optimal only when the estimate is right -- and the score is
computed from TRUE selectivities that may sit anywhere inside the stated
certified bound [S_est/F, S_est*F]. Since every step's cost is monotone
non-decreasing in each edge's own true selectivity (a bigger true value can
only inflate that step's output and every step downstream of it), the
WORST CASE over the whole uncertainty box is realized by setting every used
edge to its upper bound S_est*F. We therefore:

  1. Choose the PREFIX (start relation + first h-1 moves) that minimizes
     the WORST-CASE cost over the uncertainty box -- a minimax / robust
     reformulation, not a point-estimate plug-in. This is what keeps a
     100x-mis-estimated edge from being scheduled early "because it looks
     cheap": under the worst case it looks expensive, so it gets deferred.
  2. At the reoptimization checkpoint we do not know which bucket will be
     observed at authoring time, so for EACH bucket we independently pick
     the worst-case-robust CONTINUATION starting from that bucket's
     representative intermediate size -- a genuine contingency plan, not
     a single order replayed three times.
"""
import sys


def run_from(n, C, lo, hi, size, moves, sel, mem_cap, spill_mult):
    cost = 0.0
    for mv in moves:
        if mv == 'L':
            if lo == 0:
                return None
            edge = lo - 1
            lo -= 1
            size = size * C[lo] * sel(edge)
        else:
            if hi == n - 1:
                return None
            edge = hi
            hi += 1
            size = size * C[hi] * sel(edge)
        step_cost = size if size <= mem_cap else size * spill_mult
        cost += step_cost
    return lo, hi, size, cost


def enumerate_seqs(lo0, hi0, n, k):
    """All valid (lo,hi,moves) after k extensions from interval [lo0,hi0]."""
    stack = [(lo0, hi0, [])]
    out = []
    while stack:
        lo, hi, moves = stack.pop()
        if len(moves) == k:
            out.append((lo, hi, moves))
            continue
        if lo > 0:
            stack.append((lo - 1, hi, moves + ['L']))
        if hi < n - 1:
            stack.append((lo, hi + 1, moves + ['R']))
    return out


def main():
    itok = sys.stdin.read().split()
    idx = 0
    tid = int(itok[idx]); idx += 1
    n = int(itok[idx]); idx += 1
    C = [int(itok[idx + i]) for i in range(n)]; idx += n
    mem_cap = float(itok[idx]); idx += 1
    spill_mult = float(itok[idx]); idx += 1
    h = int(itok[idx]); idx += 1
    est, bound = [], []
    for e in range(n - 1):
        s_est = float(itok[idx]); idx += 1
        f = float(itok[idx]); idx += 1
        est.append(s_est); bound.append(f)

    def sel_worst(e):
        return min(1.0, est[e] * bound[e])

    # 1) robust PREFIX: minimize worst-case cost over start s + (h-1) moves
    best_pre = None
    for s in range(n):
        for lo, hi, moves in enumerate_seqs(s, s, n, h - 1):
            res = run_from(n, C, s, s, float(C[s]), moves, sel_worst, mem_cap, spill_mult)
            if res is None:
                continue
            _, _, size, cost = res
            if best_pre is None or cost < best_pre[0]:
                best_pre = (cost, s, moves, lo, hi, size)

    _, s, pre_moves, lo, hi, pre_size = best_pre

    # 2) robust CONTINUATION per bucket, from that bucket's representative
    #    intermediate size (worst-case-realized, matching the checker's
    #    own bucket cut points).
    reps = {"LOW": 0.5 * mem_cap, "MID": 2.0 * mem_cap, "HIGH": 8.0 * mem_cap}
    need = n - h
    branch_moves = {}
    for name, x0 in reps.items():
        best_cont = None
        for lo2, hi2, moves in enumerate_seqs(lo, hi, n, need):
            res = run_from(n, C, lo, hi, x0, moves, sel_worst, mem_cap, spill_mult)
            if res is None:
                continue
            flo, fhi, _, cost = res
            if flo != 0 or fhi != n - 1:
                continue
            if best_cont is None or cost < best_cont[0]:
                best_cont = (cost, moves)
        if best_cont is None:
            # fall back to any full-coverage continuation regardless of cost
            for lo2, hi2, moves in enumerate_seqs(lo, hi, n, need):
                if lo2 == 0 and hi2 == n - 1:
                    best_cont = (0.0, moves)
                    break
        branch_moves[name] = best_cont[1]

    out = ["START", str(s), "PRE"] + pre_moves
    out += ["BRANCH", "LOW"] + branch_moves["LOW"]
    out += ["BRANCH", "MID"] + branch_moves["MID"]
    out += ["BRANCH", "HIGH"] + branch_moves["HIGH"]
    print(" ".join(out))


if __name__ == "__main__":
    main()
