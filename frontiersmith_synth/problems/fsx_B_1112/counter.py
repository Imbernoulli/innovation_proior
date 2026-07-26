import sys
import os

# Format D checker -- minimal-multiplication shared straight-line program (SLP)
# that must correctly evaluate one fixed degree-n polynomial P at every query
# point of THREE batches (sweep / probe / ad-hoc) using as few `mul`
# instructions as possible (`add`/`sub` are free).
#
#   1) Parse the instance from <in>:
#        testId
#        n
#        a[0..n]                 (n+1 integer coefficients, a[n] != 0)
#        Q
#        q[0..Q-1]                (Q distinct integer query values, the leaf pool)
#        m1 ; idx1[0..m1-1]       sweep batch  (indices into q[])
#        m2 ; idx2[0..m2-1]       probe batch
#        m3 ; idx3[0..m3-1]       ad-hoc batch
#   2) Parse the participant SLP from <out>:
#        L
#        L lines: <op> <arg1> <arg2>    op in {mul, add, sub}
#          each arg is one of  a<k> (0<=k<=n) | q<j> (0<=j<Q) | r<i> (i < line idx)
#          -- NO free numeric literals: every leaf must be a GIVEN coefficient or
#          query value, so a submission cannot just print the target number.
#        M
#        M output register indices r<i>, in order: sweep outputs, probe outputs,
#          ad-hoc outputs (M must equal m1+m2+m3).
#   3) EXACT-equivalence gate, checked THREE times: once against the real instance
#      AND once against each of TWO independently regenerated "shadow" instances
#      that share the exact same batch/index STRUCTURE (same n, m1/m2/m3, same
#      probe repeat pattern) but fresh random coefficients/query values.  A
#      submission that happened to numerically stumble onto the right answer for
#      one specific set of numbers (e.g. via an addition-only "build the literal
#      constant" trick) will not also satisfy two independently-random shadow
#      instances, so this defense does not restrict genuinely structural
#      algorithms (Horner, finite differences, deduplication all still work,
#      because their correctness argument never depends on the specific numbers).
#   4) Objective (MINIMISE) = number of `mul` instructions = F.
#      Baseline B = (2n-1) * (m1+m2+m3)  (cost of the naive "build all powers of
#      x independently per query, then dot with coefficients" approach).
#      Ratio = min(1, 0.1 * B / F).

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen  # noqa: E402

LMAX = 20000
MAGCAP = 10 ** 80
SHADOW_SALTS = (2654435761, 40503)  # fixed, arbitrary, != main value_seed


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_ints(it, cnt):
    out = []
    for _ in range(cnt):
        out.append(int(next(it)))
    return out


def parse_operand(tok, idx, n, Q):
    if len(tok) < 2:
        return None
    head, rest = tok[0], tok[1:]
    if not rest.isdigit():
        return None
    val = int(rest)
    if head == 'a':
        return ('a', val) if 0 <= val <= n else None
    if head == 'q':
        return ('q', val) if 0 <= val < Q else None
    if head == 'r':
        return ('r', val) if 0 <= val < idx else None
    return None


def main():
    inp = open(sys.argv[1]).read().split()
    it = iter(inp)
    try:
        test_id = int(next(it))
        n = int(next(it))
        if not (1 <= n <= 4096):
            fail("bad n")
        a = read_ints(it, n + 1)
        Q = int(next(it))
        if not (1 <= Q <= 200000):
            fail("bad Q")
        q = read_ints(it, Q)
        m1 = int(next(it)); idx1 = read_ints(it, m1)
        m2 = int(next(it)); idx2 = read_ints(it, m2)
        m3 = int(next(it)); idx3 = read_ints(it, m3)
    except (StopIteration, ValueError):
        fail("malformed instance")
        return
    if a[n] == 0:
        fail("degenerate instance")
    for lst in (idx1, idx2, idx3):
        for v in lst:
            if not (0 <= v < Q):
                fail("instance index out of range")
    M = m1 + m2 + m3

    # ---- parse participant SLP ----
    try:
        toks = open(sys.argv[2]).read().split()
    except OSError:
        fail("cannot read output")
        return
    if not toks:
        fail("empty output")
        return
    oit = iter(toks)
    try:
        L = int(next(oit))
    except (StopIteration, ValueError):
        fail("bad L")
        return
    if not (1 <= L <= LMAX):
        fail("L out of range")
        return

    instrs = []
    try:
        for idx in range(L):
            op = next(oit)
            t1 = next(oit)
            t2 = next(oit)
            if op not in ('mul', 'add', 'sub'):
                fail("bad op at %d" % idx); return
            arg1 = parse_operand(t1, idx, n, Q)
            arg2 = parse_operand(t2, idx, n, Q)
            if arg1 is None or arg2 is None:
                fail("bad operand at %d" % idx); return
            instrs.append((op, arg1, arg2))
    except StopIteration:
        fail("truncated program"); return

    try:
        M_out = int(next(oit))
    except (StopIteration, ValueError):
        fail("bad output count"); return
    if M_out != M:
        fail("output count %d != expected %d" % (M_out, M)); return
    out_regs = []
    try:
        for _ in range(M_out):
            r = int(next(oit))
            if not (0 <= r < L):
                fail("output register out of range"); return
            out_regs.append(r)
    except (StopIteration, ValueError):
        fail("bad output register"); return

    mulcount = sum(1 for (op, _, _) in instrs if op == 'mul')
    if mulcount == 0:
        fail("no multiplications (cannot realise a degree>=1 polynomial)")
        return

    def evaluate(a_vals, q_vals):
        regs = []
        for (op, arg1, arg2) in instrs:
            def rv(arg):
                kind, val = arg
                if kind == 'a':
                    return a_vals[val]
                if kind == 'q':
                    return q_vals[val]
                return regs[val]
            v1, v2 = rv(arg1), rv(arg2)
            if op == 'mul':
                v = v1 * v2
            elif op == 'add':
                v = v1 + v2
            else:
                v = v1 - v2
            if v > MAGCAP or v < -MAGCAP:
                return None
            regs.append(v)
        return regs

    def horner(a_vals, x):
        acc = 0
        for c in reversed(a_vals):
            acc = acc * x + c
        return acc

    def targets(a_vals, q_vals):
        return ([horner(a_vals, q_vals[i]) for i in idx1] +
                [horner(a_vals, q_vals[i]) for i in idx2] +
                [horner(a_vals, q_vals[i]) for i in idx3])

    def check_instance(a_vals, q_vals):
        regs = evaluate(a_vals, q_vals)
        if regs is None:
            return False
        tgt = targets(a_vals, q_vals)
        for r, want in zip(out_regs, tgt):
            if regs[r] != want:
                return False
        return True

    if not check_instance(a, q):
        fail("program does not compute the target on the given instance")
        return

    for salt in SHADOW_SALTS:
        shadow = gen.build(test_id, value_seed=salt * (test_id + 1) + 17)
        if shadow["n"] != n or shadow["Q"] != Q or shadow["m1"] != m1 or \
           shadow["m2"] != m2 or shadow["m3"] != m3 or \
           shadow["idx1"] != idx1 or shadow["idx2"] != idx2 or shadow["idx3"] != idx3:
            fail("internal shadow-structure mismatch")
            return
        if not check_instance(shadow["a"], shadow["q"]):
            fail("program fails on an independently regenerated shadow instance "
                 "(same structure, different numbers) -- looks like a numeric "
                 "coincidence rather than a genuine computation")
            return

    B = (2 * n - 1) * M
    F = mulcount
    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    print("muls=%d baseline=%d M=%d Ratio: %.6f" % (F, B, M, sc / 1000.0))


if __name__ == "__main__":
    main()
