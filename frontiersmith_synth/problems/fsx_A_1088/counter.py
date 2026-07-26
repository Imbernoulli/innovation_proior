#!/usr/bin/env python3
"""counter.py <in> <out> <ans> -- Format D checker for "Cheapest Clockwork" (fsx_A_1088).

<in>:  n
       k
       s_1 ... s_k      (the pointwise-fixed set S, k values in [0,n))
       L                (required minimum cycle length for the orbit of 0)

<out>: m                                (number of straight-line instructions, 1<=m<=MAX_OPS)
       m lines, each one of:
         ADD  a b d      r[d] = (r[a] + r[b])  mod n
         MUL  a b d      r[d] = (r[a] * r[b])  mod n
         ADDC a K d      r[d] = (r[a] + K)     mod n
         MULC a K d      r[d] = (r[a] * K)     mod n
       a,b,d are register indices in [0,NUM_REGS); K is an integer constant in [0,n).
       Register r0 starts holding the input x; every other register starts at 0.
       The submitted function is f(x) := value of r0 after all m instructions run.

Feasibility (checked by EXACT simulation over every x in [0,n), vectorized):
  1. f must be a permutation of {0,...,n-1}.
  2. f(s) == s for every s in S.
  3. The cycle of the permutation containing 0 must have length >= L.
Any violation -> "Ratio: 0.0". Otherwise the objective is the instruction count m
(minimize); the checker's own fixed reference op-budget B trades off against it:
    ratio = min(1, 100*B/(1000*m))
"""
import sys

try:
    import numpy as np
except Exception:
    np = None

MAX_OPS = 150
NUM_REGS = 8
BASELINE_OPS = 15.0
VALID_OPS = {"ADD", "MUL", "ADDC", "MULC"}


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    toks = open(path).read().split()
    idx = 0
    n = int(toks[idx]); idx += 1
    k = int(toks[idx]); idx += 1
    S = [int(toks[idx + i]) for i in range(k)]; idx += k
    L = int(toks[idx]); idx += 1
    return n, k, S, L


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    in_path, out_path = sys.argv[1], sys.argv[2]

    n, k, S, L = read_instance(in_path)
    if n < 2:
        fail("degenerate instance")

    try:
        with open(out_path, "r", errors="replace") as f:
            raw_lines = [ln.strip() for ln in f]
    except Exception:
        fail("cannot read output")
    lines = [ln for ln in raw_lines if ln != ""]
    if not lines:
        fail("empty output")

    try:
        m = int(lines[0])
    except ValueError:
        fail("op-count header is not an integer")
    if m < 1 or m > MAX_OPS:
        fail("op count %d out of range [1,%d]" % (m, MAX_OPS))
    if len(lines) < 1 + m:
        fail("fewer than %d instruction lines present" % m)

    instrs = []
    for i in range(m):
        parts = lines[1 + i].split()
        if len(parts) != 4:
            fail("instruction %d does not have exactly 4 tokens" % i)
        op = parts[0]
        if op not in VALID_OPS:
            fail("instruction %d has unknown opcode %r" % (i, op))
        try:
            a = int(parts[1])
            bK = int(parts[2])
            d = int(parts[3])
        except ValueError:
            fail("instruction %d has a non-integer operand" % i)
        if not (0 <= a < NUM_REGS) or not (0 <= d < NUM_REGS):
            fail("instruction %d references an out-of-range register" % i)
        if op in ("ADD", "MUL"):
            if not (0 <= bK < NUM_REGS):
                fail("instruction %d references an out-of-range register" % i)
        else:
            if not (0 <= bK < n):
                fail("instruction %d constant %d out of range [0,%d)" % (i, bK, n))
        instrs.append((op, a, bK, d))

    # ---- exact simulation over the whole domain, vectorized across x ----
    if np is not None:
        xs = np.arange(n, dtype=np.int64)
        regs = [np.zeros(n, dtype=np.int64) for _ in range(NUM_REGS)]
        regs[0] = xs
        nn = int(n)
        for op, a, bK, d in instrs:
            if op == "ADD":
                regs[d] = (regs[a] + regs[bK]) % nn
            elif op == "MUL":
                regs[d] = (regs[a] * regs[bK]) % nn
            elif op == "ADDC":
                regs[d] = (regs[a] + bK) % nn
            else:  # MULC
                regs[d] = (regs[a] * bK) % nn
        fvals = regs[0]
        if int(fvals.min()) < 0 or int(fvals.max()) >= n:
            fail("output value out of range")
        if len(np.unique(fvals)) != n:
            fail("f is not a permutation of Z_n (not bijective)")
        fmap = [int(v) for v in fvals]
    else:
        fmap = list(range(n))
        for x in range(n):
            regs = [0] * NUM_REGS
            regs[0] = x
            for op, a, bK, d in instrs:
                if op == "ADD":
                    regs[d] = (regs[a] + regs[bK]) % n
                elif op == "MUL":
                    regs[d] = (regs[a] * regs[bK]) % n
                elif op == "ADDC":
                    regs[d] = (regs[a] + bK) % n
                else:
                    regs[d] = (regs[a] * bK) % n
            fmap[x] = regs[0]
        if len(set(fmap)) != n:
            fail("f is not a permutation of Z_n (not bijective)")

    for s in S:
        if not (0 <= s < n):
            fail("S element %d out of range (bad input)" % s)
        if fmap[s] != s:
            fail("f does not fix required point %d (f(%d)=%d)" % (s, s, fmap[s]))

    y = fmap[0]
    steps = 1
    limit = 2 * n + 5
    while y != 0:
        y = fmap[y]
        steps += 1
        if steps > limit:
            fail("cycle containing 0 did not close (should be impossible for a permutation)")
    if steps < L:
        fail("cycle length %d < required L=%d" % (steps, L))

    ops = float(m)
    B = BASELINE_OPS
    sc = min(1000.0, 100.0 * B / max(1e-9, ops))
    print("cycle_len=%d ops=%d baseline=%.1f" % (steps, m, B))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == "__main__":
    main()
