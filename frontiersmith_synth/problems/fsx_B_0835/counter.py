#!/usr/bin/env python3
"""
counter.py <in> <out> <ans>  -> prints "... Ratio: <float>" on its last line.

Verifies the submitted decision program (a) is well-formed, (b) is acyclic and
terminates within a step cap on every patient, (c) outputs the CORRECT given
label for every patient, then computes the EXACT (Fraction) population-weighted
expected op count, where op count on a root-to-leaf path = (# distinct new
prep instructions executed, tracking a "live" set that persists along the
path) + (1 compare per test node). Normalizes against a full-scan baseline
that unconditionally runs every test (in the given order) on every patient.
"""
import sys
from fractions import Fraction


def resolve(tok, F, vals):
    if tok[0] == 'F':
        return F[int(tok[1:])]
    if tok[0] == 'I':
        return vals[int(tok[1:])]
    if tok[0] == 'C':
        return int(tok[1:])
    raise ValueError(tok)


def eval_all(instrs, F):
    vals = []
    for (op, a, b) in instrs:
        va, vb = resolve(a, F, vals), resolve(b, F, vals)
        if op == 'ADD':
            v = va + vb
        elif op == 'SUB':
            v = va - vb
        elif op == 'MUL':
            v = va * vb
        else:
            raise ValueError(op)
        vals.append(v)
    return vals


def closure_of(instrs, final_idx):
    seen = set()
    stack = [final_idx]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        op, a, b = instrs[idx]
        for tok in (a, b):
            if tok[0] == 'I':
                j = int(tok[1:])
                if j not in seen:
                    stack.append(j)
    return seen


def fail(reason):
    print("INFEASIBLE: %s Ratio: 0.0" % reason)
    sys.exit(0)


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path) as f:
        in_tokens = f.read().split()
    p = 0
    def nxt():
        nonlocal p
        v = in_tokens[p]
        p += 1
        return v

    K = int(nxt()); M = int(nxt()); T = int(nxt()); N = int(nxt())
    instrs = []
    for _ in range(M):
        op = nxt(); a = nxt(); b = nxt()
        if op not in ('ADD', 'SUB', 'MUL'):
            fail("bad opcode in input (should not happen)")
        instrs.append((op, a, b))
    tests = []
    for _ in range(T):
        fin = int(nxt()); thr = int(nxt())
        tests.append((fin, thr))
    patients = []
    total_w = 0
    for _ in range(N):
        F = [int(nxt()) for _ in range(K)]
        w = int(nxt())
        lab = int(nxt())
        patients.append((F, w, lab))
        total_w += w

    if total_w <= 0:
        fail("degenerate input weight (should not happen)")

    # precompute per-test dependency closures (structural, patient-independent)
    test_closure = [closure_of(instrs, fin) for (fin, thr) in tests]

    # precompute per-test outcome bit for every patient
    outcome = []  # outcome[i][t] in {0,1}
    for (F, w, lab) in patients:
        vals = eval_all(instrs, F)
        bits = []
        for (fin, thr) in tests:
            bits.append(1 if vals[fin] >= thr else 0)
        outcome.append(bits)

    # ---- read participant's decision program ----
    try:
        with open(out_path) as f:
            out_tokens = f.read().split()
    except FileNotFoundError:
        fail("missing output file")

    if not out_tokens:
        fail("empty output")

    def is_int_tok(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    q = 0
    if not is_int_tok(out_tokens[q]):
        fail("first token must be an integer node count (rejects nan/inf/garbage)")
    numnodes = int(out_tokens[q]); q += 1
    MAX_NODES = 4000
    if numnodes <= 0 or numnodes > MAX_NODES:
        fail("node count out of range")

    nodes = []  # each: ('LEAF', label) or ('TEST', testIdx, lo, hi)
    for i in range(numnodes):
        if q >= len(out_tokens):
            fail("truncated output")
        kind = out_tokens[q]; q += 1
        if kind == 'LEAF':
            if q >= len(out_tokens) or not is_int_tok(out_tokens[q]):
                fail("bad LEAF token (non-finite or missing)")
            lab = int(out_tokens[q]); q += 1
            nodes.append(('LEAF', lab))
        elif kind == 'TEST':
            if q + 2 >= len(out_tokens):
                fail("truncated TEST node")
            toks = out_tokens[q:q + 3]
            if not all(is_int_tok(t) for t in toks):
                fail("bad TEST token (non-finite or missing)")
            ti, lo, hi = int(toks[0]), int(toks[1]), int(toks[2])
            q += 3
            if not (0 <= ti < T):
                fail("test index out of range")
            if not (0 <= lo < numnodes) or not (0 <= hi < numnodes):
                fail("child node index out of range")
            nodes.append(('TEST', ti, lo, hi))
        else:
            fail("unknown node kind token")

    if q != len(out_tokens):
        fail("trailing garbage after declared nodes")

    MAX_STEPS = 2 * T + 8

    total_cost_weighted = 0  # exact integer (sum of weight_i * cost_i)
    for pi, (F, w, lab) in enumerate(patients):
        live = set()
        cost = 0
        cur = 0
        steps = 0
        while True:
            steps += 1
            if steps > MAX_STEPS:
                fail("decision program did not terminate within step cap (cycle?)")
            node = nodes[cur]
            if node[0] == 'LEAF':
                if node[1] != lab:
                    fail("wrong label for a patient (id=%d)" % pi)
                break
            _, ti, lo, hi = node
            clos = test_closure[ti]
            new_instrs = clos - live
            cost += len(new_instrs) + 1
            live |= clos
            bit = outcome[pi][ti]
            cur = hi if bit == 1 else lo
        total_cost_weighted += w * cost

    F_exact = Fraction(total_cost_weighted, total_w)

    # ---- baseline: run ALL tests, in the given fixed order, on every patient ----
    baseline_union = set()
    for clos in test_closure:
        baseline_union |= clos
    B = len(baseline_union) + T

    if F_exact <= 0:
        fail("nonpositive cost (should not happen)")

    sc = Fraction(100 * B, 1) / F_exact
    sc = min(Fraction(1000, 1), sc)
    ratio = float(sc) / 1000.0
    print("Correct decision program. baseline=%d expected_cost=%s Ratio: %.6f" % (B, str(F_exact), ratio))


if __name__ == "__main__":
    main()
