#!/usr/bin/env python3
# gen.py -- family: adaptive-sparsity-probe-plan (Format D, minimize op-count)
#
# `python3 gen.py <testId>` prints ONE instance to stdout.  Seeded by testId only.
#
# An instance is a block-computation over B blocks and a published SWEEP of P
# sparsity patterns.  Under pattern q, block j is either ZERO (contributes
# nothing) or NONZERO (its multiply MUST be performed).  The solver compiles ONE
# branching plan (a decision DAG over the ops TEST/MUL/HALT) that, executed on
# each of the P patterns, performs every nonzero block's multiply; the score is
# the WORST-CASE exact op-cost of that single plan over the whole sweep.
#
# Planted structure (the "innovation hook"):  a small set of DISCRIMINATOR blocks
# whose zero/nonzero values across the sweep encode each pattern's identity, plus
# a per-pattern PAYLOAD of content blocks drawn from a shared pool.  Two patterns
# never share the same nonzero-set.  The block columns are permuted so the
# discriminators are hidden -- the solver must DISCOVER the cheapest splitting
# invariant from the pattern list itself.
#
# Output format (stdin the solver reads):
#   line 1:  B P M      (B blocks; P patterns; M = op-cost of one block multiply)
#   next P lines:  a length-B string over {'0','1'};  '1' = block nonzero (must
#                  be multiplied), '0' = block zero.
import sys, random

def build(testId):
    t = int(testId)
    rng = random.Random(90210 + 17 * t)

    P = 8 + t                       # 9 .. 18 patterns
    d = max(1, (P - 1).bit_length())  # discriminator blocks needed to encode identity
    s = 3                           # payload size (content blocks per pattern)
    C = s * P                       # content-pool size (disjoint payloads -> full union)
    M = 14                          # op-cost of one block multiply (a TEST costs 1)

    B = d + C
    pool = list(range(d, d + C))    # content-block indices (original numbering)
    rng.shuffle(pool)               # scramble which content block lands in which payload

    # Each pattern p has a DISTINCT identity code (0..P-1).  Discriminator block k
    # is nonzero in pattern p iff bit k of the code is set.  Payload = a DISJOINT
    # chunk of s content blocks, so the union of all payloads is the whole pool
    # (large baseline) while any single pattern touches only s of them.
    nz_sets = []
    for p in range(P):
        nz = set()
        for k in range(d):
            if (p >> k) & 1:
                nz.add(k)               # discriminator block
        nz |= set(pool[p * s:(p + 1) * s])
        nz_sets.append(nz)

    # Hide the discriminators: permute block columns.
    perm = list(range(B))
    rng.shuffle(perm)                    # original index i -> output column perm[i]

    lines = ["%d %d %d" % (B, P, M)]
    for p in range(P):
        bits = ['0'] * B
        for i in nz_sets[p]:
            bits[perm[i]] = '1'
        lines.append("".join(bits))
    return "\n".join(lines) + "\n"

if __name__ == "__main__":
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    sys.stdout.write(build(tid))
