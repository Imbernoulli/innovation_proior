#!/usr/bin/env python3
# counter.py -- Format D checker for family adaptive-sparsity-probe-plan.
#
# Input (<in>):
#   line 1:  B P M
#   next P lines: length-B strings over {'0','1'} (pattern nonzero masks).
#
# Output (<out>):  a branching plan = a decision DAG.
#   line 1:  N               (number of nodes, 1..300000; node 0 is the START)
#   next N lines, node i:
#       T j a b     TEST block j (cost 1); go to node a if block j is ZERO in the
#                   current pattern, else to node b.
#       M i c       MULTIPLY block i (cost M); then go to node c.
#       H           HALT.
#   Execution starts at node 0; the pattern acts as the oracle answering TESTs.
#
# FEASIBILITY (strict; any violation -> Ratio 0.0):
#   * well-formed schema, integer tokens, indices in range, finite;
#   * every execution path terminates (no cycle) within N+2 steps;
#   * for EVERY pattern, the multiplies performed on its path cover ALL of that
#     pattern's nonzero blocks.
#
# OBJECTIVE (minimize):  F = max over patterns of (#TESTs + M * #MULs) on its path.
# BASELINE B_ck = M * |union of all nonzero blocks|  (the static "multiply the
# union, no tests" plan the checker can always build itself).
#   Ratio = min(1.0, 0.1 * B_ck / F).   trivial-union -> 0.1;  10x better caps at 1.
import sys

def out_ratio(x, note=""):
    if note:
        print("info: %s" % note)
    print("Ratio: %.6f" % x)
    sys.exit(0)

def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)

def main():
    try:
        raw_in = open(sys.argv[1]).read().split()
        raw_out_lines = open(sys.argv[2]).read().splitlines()
    except Exception:
        fail("io")

    # ---- parse instance ----
    it = iter(raw_in)
    try:
        B = int(next(it)); P = int(next(it)); M = int(next(it))
    except Exception:
        fail("bad header")
    if not (1 <= B <= 100000) or not (1 <= P <= 100000) or not (1 <= M <= 10**9):
        fail("bad header ranges")
    patterns = []       # list of bytearray-like bool lists
    nz_sets = []
    union = set()
    try:
        for _ in range(P):
            row = next(it)
            if len(row) != B or any(ch not in "01" for ch in row):
                fail("bad pattern row")
            nz = [c == '1' for c in row]
            s = set(i for i, v in enumerate(nz) if v)
            patterns.append(nz)
            nz_sets.append(s)
            union |= s
    except StopIteration:
        fail("missing pattern rows")

    if not union:
        fail("degenerate instance (no nonzero blocks)")

    # ---- parse plan ----
    lines = [ln.strip() for ln in raw_out_lines if ln.strip() != ""]
    if not lines:
        fail("empty plan")
    try:
        N = int(lines[0])
    except Exception:
        fail("bad N")
    if not (1 <= N <= 300000):
        fail("bad N range")
    if len(lines) - 1 < N:
        fail("too few node lines")

    nodes = []  # tuples: ('H',) | ('T', j, a, b) | ('M', i, c)
    for t in range(N):
        parts = lines[1 + t].split()
        if not parts:
            fail("blank node")
        typ = parts[0]
        try:
            if typ == 'H':
                if len(parts) != 1:
                    fail("bad H arity")
                nodes.append(('H',))
            elif typ == 'T':
                if len(parts) != 4:
                    fail("bad T arity")
                j = int(parts[1]); a = int(parts[2]); b = int(parts[3])
                if not (0 <= j < B) or not (0 <= a < N) or not (0 <= b < N):
                    fail("T index out of range")
                nodes.append(('T', j, a, b))
            elif typ == 'M':
                if len(parts) != 3:
                    fail("bad M arity")
                i = int(parts[1]); c = int(parts[2])
                if not (0 <= i < B) or not (0 <= c < N):
                    fail("M index out of range")
                nodes.append(('M', i, c))
            else:
                fail("unknown node type")
        except ValueError:
            fail("non-integer token")

    # ---- execute plan on every pattern ----
    step_cap = N + 2
    worst = 0
    for p in range(P):
        nz = patterns[p]
        cur = 0
        steps = 0
        tests = 0
        muls = 0
        multiplied = set()
        while True:
            steps += 1
            if steps > step_cap:
                fail("plan does not terminate (cycle) on a pattern")
            nd = nodes[cur]
            k = nd[0]
            if k == 'H':
                break
            elif k == 'T':
                tests += 1
                cur = nd[2] if not nz[nd[1]] else nd[3]
            else:  # 'M'
                muls += 1
                multiplied.add(nd[1])
                cur = nd[2]
        # coverage: all nonzero blocks of this pattern must be multiplied
        if not nz_sets[p] <= multiplied:
            fail("plan fails to compute a required nonzero block")
        cost = tests + M * muls
        if cost > worst:
            worst = cost

    F = max(1, worst)
    B_ck = M * len(union)
    sc = min(1000.0, 100.0 * B_ck / F)
    out_ratio(sc / 1000.0, "F=%d B_ck=%d union=%d" % (F, B_ck, len(union)))

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        fail("exception:%s" % type(e).__name__)
