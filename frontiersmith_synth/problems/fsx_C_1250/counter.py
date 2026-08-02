#!/usr/bin/env python3
"""
counter.py -- Format D checker for "Multiplying Almost Right to Save Power".

Reads the instance (K, G, S, TMAX, COMP_EXTRA, area table, per-chain length+budget,
per-position sample operand pairs) from <in>, and the participant's per-position
truncation configuration from <out>:

    K lines, line i (1-indexed, matching input position order): "t_i c_i"
      0 <= t_i <= TMAX     (bits of each operand to clear)
      c_i in {0,1}         (0 = floor truncation, 1 = round-to-nearest / compensated)

Feasibility: for every chain g, over every one of its S sample vectors, the chain's
approximated MAC sum (using each member position's own t_i,c_i) must differ from the
exact sum by at most the chain's budget B_g. Any parse failure, out-of-range token,
non-finite token, wrong token count, or budget violation -> Ratio: 0.0.

Objective (minimize): total area = sum_i area_table[t_i] + (COMP_EXTRA if c_i==1 else 0).
Baseline B (checker's own trivial construction) = every position at t=0, c=0 (exact
multiplication, area_table[0] each) -- always feasible since it has zero error.

    Ratio = min(1000, 100 * B / F) / 1000   (fewer-area-than-baseline saturates at 1.0)
"""
import sys


def trunc0(x, t):
    return x if t == 0 else (x >> t) << t


def trunc1(x, t):
    return x if t == 0 else ((x + (1 << (t - 1))) >> t) << t


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    in_toks = open(sys.argv[1]).read().split()
    out_text = open(sys.argv[2]).read()

    it = iter(in_toks)
    try:
        K = int(next(it)); G = int(next(it)); S = int(next(it))
        TMAX = int(next(it)); COMP_EXTRA = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= K <= 100000 and 1 <= G <= K and 1 <= S <= 1000 and 0 <= TMAX <= 30):
        fail("bad header ranges")

    try:
        area = [int(next(it)) for _ in range(TMAX + 1)]
    except Exception:
        fail("bad area table")
    for t in range(TMAX):
        if not (area[t] > area[t + 1] > 0):
            fail("area table not strictly decreasing / positive")

    chain_len = []
    chain_budget = []
    try:
        for _ in range(G):
            L = int(next(it)); Bg = int(next(it))
            if L < 1 or Bg < 0:
                fail("bad chain header")
            chain_len.append(L)
            chain_budget.append(Bg)
    except Exception:
        fail("bad chain headers")
    if sum(chain_len) != K:
        fail("chain lengths do not sum to K")

    # positions[i] = list of (a,b) for s=0..S-1 ; chain_of[i] = chain index
    positions = []
    chain_of = []
    try:
        for gi, L in enumerate(chain_len):
            for _ in range(L):
                pairs = []
                for _ in range(S):
                    a = int(next(it)); b = int(next(it))
                    if a < 0 or b < 0:
                        fail("negative operand in instance")
                    pairs.append((a, b))
                positions.append(pairs)
                chain_of.append(gi)
    except Exception:
        fail("bad position sample data")

    # ---- parse participant output: exactly K lines of "t_i c_i" ----
    out_toks = out_text.split()
    if len(out_toks) != 2 * K:
        fail("wrong token count (got %d, need %d)" % (len(out_toks), 2 * K))
    cfg_t = [0] * K
    cfg_c = [0] * K
    try:
        for i in range(K):
            tt = int(out_toks[2 * i])
            cc = int(out_toks[2 * i + 1])
            cfg_t[i] = tt
            cfg_c[i] = cc
    except Exception:
        fail("non-integer token (or nan/inf) in output")

    for i in range(K):
        if not (0 <= cfg_t[i] <= TMAX):
            fail("t_%d out of range" % i)
        if cfg_c[i] not in (0, 1):
            fail("c_%d not in {0,1}" % i)

    # ---- feasibility: per-chain max absolute error over all S samples <= budget ----
    chain_exact = [[0] * S for _ in range(G)]
    chain_approx = [[0] * S for _ in range(G)]
    for i, pairs in enumerate(positions):
        gi = chain_of[i]
        t = cfg_t[i]; c = cfg_c[i]
        for s, (a, b) in enumerate(pairs):
            chain_exact[gi][s] += a * b
            if c == 0:
                chain_approx[gi][s] += trunc0(a, t) * trunc0(b, t)
            else:
                chain_approx[gi][s] += trunc1(a, t) * trunc1(b, t)

    for gi in range(G):
        worst = max(abs(chain_exact[gi][s] - chain_approx[gi][s]) for s in range(S))
        if worst > chain_budget[gi]:
            fail("chain %d exceeds budget (%d > %d)" % (gi, worst, chain_budget[gi]))

    # ---- objective ----
    F = 0
    for i in range(K):
        F += area[cfg_t[i]] + (COMP_EXTRA if cfg_c[i] == 1 else 0)
    B = K * area[0]

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("K=%d F=%d B=%d Ratio: %.6f" % (K, F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
