#!/usr/bin/env python3
"""
gen.py <testId> -- print ONE query-plan-choose instance to stdout.

A CHAIN join query over n base relations R_0..R_{n-1} (relation i and i+1
share a join predicate, i = 0..n-2).  Base cardinalities C_i are EXACT
(known table stats). Each join edge e has an ESTIMATED selectivity S_est[e]
and a CERTIFIED multiplicative uncertainty bound F[e] >= 1: the TRUE
selectivity is guaranteed to lie in [S_est/F, S_est*F] (clipped to (0,1]),
but its exact value is NEVER revealed here -- it lives only inside the
checker, seeded by the test id (see verify.py).

Executing a plan means repeatedly extending a contiguous interval of the
chain (left or right) starting from a chosen relation; each extension's
output cardinality is the running size times the next relation's
cardinality times that edge's selectivity.  If an extension's output
exceeds MEM_CAP rows, the executor spills to disk and that step's cost is
multiplied by SPILL_MULT.

At a fixed REOPTIMIZATION CHECKPOINT (after `h` relations are joined), the
executor can observe the intermediate result's TRUE size and branch the
remainder of the plan on which bucket (LOW/MID/HIGH, relative to MEM_CAP)
it falls into -- this is the adaptive-reoptimization point.

7 of the 10 tests plant one edge with a huge (100x) uncertainty bound
whose true selectivity the checker deterministically realizes at the
EXPENSIVE extreme -- exactly where a plan built from the point estimate
alone pivots badly.  Everything is seeded by testId only.
"""
import sys, random


def spec_for(tid):
    n = 5 + (tid - 1) % 5                       # 5..9, small scale
    h = min(max(2, n // 2), n - 1)               # 2 <= h <= n-1
    trap = tid in (2, 3, 4, 5, 7, 8, 10)         # 7/10 tests carry a big trap
    return n, h, trap


def main():
    tid = int(sys.argv[1])
    n, h, trap = spec_for(tid)
    rng = random.Random(913001 + 7919 * tid)

    C = [rng.randint(300, 3000) for _ in range(n)]
    avg_c = sum(C) / len(C)

    trap_edge = rng.randrange(0, n - 1) if trap else -1

    est, bound = [], []
    for e in range(n - 1):
        if e == trap_edge:
            s_est = round(rng.uniform(0.00005, 0.00025), 6)
            f = 100.0
        else:
            s_est = round(rng.uniform(0.00030, 0.00120), 6)
            f = round(rng.uniform(1.15, 1.8), 3)
        est.append(s_est)
        bound.append(f)

    mem_cap = int(round(0.9 * avg_c))
    spill_mult = 5.0

    lines = [str(tid), str(n), " ".join(str(c) for c in C),
              "%d %.4f" % (mem_cap, spill_mult), str(h)]
    for e in range(n - 1):
        lines.append("%.6f %.4f" % (est[e], bound[e]))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
