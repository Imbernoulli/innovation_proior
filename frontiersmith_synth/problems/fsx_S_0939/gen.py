#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy training probe table to stdout.

A mystery 4-opcode CPU (ADD/MUL/LOAD/STORE) executes short straight-line
programs in order, one issue slot per cycle, except when hazards stall it.
Each probe program is reduced to eight structural counts (never the raw
opcode/dependency sequence) plus a stopwatch-measured cycle count.

The TRAIN table the solver sees is drawn from a regime where consecutive-MUL
clusters are always SHORT (at most a couple of adjacent multiplies).  The
HELD-OUT grading probes (regenerated only inside the grader) contain much
LONGER consecutive-MUL clusters -- a regime never shown here.  The hidden
per-hazard costs, the exact shape of the multiply-unit contention law, and
all RNG seeds are NEVER printed.  STDOUT prints ONLY the data table.
"""
import sys, random

# ---- fixed design constants (mirrored byte-for-byte in verify.py) ----
T_TRAIN = 100
N_TRAIN_LO, N_TRAIN_HI = 10, 26
MAXR_TRAIN = 3
P_DEP_TRAIN = 0.5
NOISE_TRAIN_SIGMA = 1.2


def true_cycles(n, cLU, cMF, cST):
    """Hidden mechanistic law (identical in gen.py and verify.py); never printed."""
    return n + 2 * cLU + cMF + cST * (cST + 1) // 2


def gen_program(rng, n, maxR, p_dep):
    """Sample one straight-line program's structural counts (opcodes A/M/L/S,
    a single contiguous MUL cluster of length 0..maxR, RAW-dependency bits on
    every other adjacent pair)."""
    R = rng.randint(0, maxR)
    if R > n:
        R = n
    op = [None] * n
    if R > 0:
        start = rng.randint(0, n - R)
        for i in range(start, start + R):
            op[i] = 'M'
    for i in range(n):
        if op[i] is None:
            op[i] = rng.choice(('A', 'L', 'S'))
    dep = [0] * n
    for j in range(1, n):
        prev, cur = op[j - 1], op[j]
        if prev == 'S':
            dep[j] = 0
        elif prev == 'M' and cur == 'M':
            dep[j] = 0
        else:
            dep[j] = 1 if rng.random() < p_dep else 0
    nA = op.count('A'); nM = op.count('M'); nL = op.count('L'); nS = op.count('S')
    cLU = sum(1 for j in range(1, n) if dep[j] == 1 and op[j - 1] == 'L')
    cMF = sum(1 for j in range(1, n) if dep[j] == 1 and op[j - 1] == 'M')
    cST = sum(1 for j in range(1, n) if op[j - 1] == 'M' and op[j] == 'M')
    return n, nA, nM, nL, nS, cLU, cMF, cST


def gen_train(t):
    rng = random.Random(31 + t * 104729)
    rows = []
    for _ in range(T_TRAIN):
        n = rng.randint(N_TRAIN_LO, N_TRAIN_HI)
        n_, nA, nM, nL, nS, cLU, cMF, cST = gen_program(rng, n, MAXR_TRAIN, P_DEP_TRAIN)
        cyc = true_cycles(n_, cLU, cMF, cST) + rng.gauss(0.0, NOISE_TRAIN_SIGMA)
        rows.append((n_, nA, nM, nL, nS, cLU, cMF, cST, cyc))
    return rows


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = gen_train(t)
    out = ["%d %d" % (len(rows), t)]
    for n, nA, nM, nL, nS, cLU, cMF, cST, cyc in rows:
        out.append("%d %d %d %d %d %d %d %d %.3f" % (n, nA, nM, nL, nS, cLU, cMF, cST, cyc))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
