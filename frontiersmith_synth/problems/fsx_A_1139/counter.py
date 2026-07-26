import sys

# Format D checker -- straight-line-program (DAG) evaluator for a length-n NTT.
#
# Instance <in>:  "n q w"   (w is a primitive n-th root of unity mod prime q).
#
# Participant <out> defines registers 0..n-1 as the (implicit) input x[0..n-1] and
# then a straight-line program of L instructions; instruction i (0-indexed) defines
# register n+i as one of:
#   A a b   reg[n+i] = reg[a] + reg[b]           (mod q)   -- CHEAP (costs ADD_COST)
#   S a b   reg[n+i] = reg[a] - reg[b]           (mod q)   -- CHEAP (costs ADD_COST)
#   M a c   reg[n+i] = reg[a] * c                (mod q)   -- counts as 1 general
#                                                              multiplication UNLESS
#                                                              c mod q in {0,1,q-1}
# (only additions/subtractions of two DATA registers, or multiplication of a DATA
#  register by an explicit FIELD CONSTANT, are permitted -- no data*data products --
#  so the whole program is manifestly LINEAR in the input; checking it against the n
#  standard basis vectors therefore verifies it against EVERY input exactly).
#
# ADD_COST is deliberately NOT zero: over a field, "x+x" is exactly a multiplication
# of x by the constant 2, and any constant c can be reached from x by a double-and-add
# chain of O(log c) free additions. If additions were free, EVERY multiplication could
# be laundered into an addition chain for an unearned Ratio: 1.0. Charging a small
# ADD_COST per addition (the format brief explicitly allows "adds free OR CHEAP") makes
# laundering a multiplication into a ~15-40-step addition chain strictly more expensive
# than just paying for the multiplication, while remaining negligible next to the
# genuine multiplication counts that separate the trivial/greedy/strong tiers.
#
# Format:
#   R L
#   <L instruction lines>
#   O o_0 o_1 ... o_{n-1}      (register ids holding X[0..n-1])
#
# Feasibility: for every j in [0,n), running the program with input = e_j (the j-th
# standard basis vector) must reproduce column j of the true NTT matrix EXACTLY:
#   expected X[k] = w^(j*k mod n) mod q.
# Any parse error, out-of-range/non-causal register reference, non-integer/garbage
# token, or mismatch -> Ratio: 0.0.
#
# Objective (minimize) F = mult_count + ADD_COST * add_count, where mult_count counts
# "M" instructions with a non-trivial constant (not in {0,1,q-1}) and add_count counts
# all "A"/"S" instructions.
# Baseline B = number of nontrivial entries of the naive n x n NTT matrix (the
# textbook "one multiplication per matrix entry" construction achieves exactly B
# multiplications, so B alone -- not B's own add-chain cost -- is the reference).
#   Ratio = min(1, SCALE * B / F)
# SCALE is rescaled DOWN from the format's usual 0.1 because for these instance sizes
# the naive O(n^2) baseline grows much faster than the reachable multiplicative
# complexity, so 0.1 would let the strong reference saturate at Ratio=1.0 on the
# larger/three-factor instances; SCALE=0.05 keeps every instance (including the
# hardest, n=105) safely below the required 0.92 ceiling while still leaving the
# naive baseline (trivial) inside the calibrated [0.03,0.35] baseline band.

SCALE = 0.05
ADD_COST = 0.1
MAXTOK = 30
MAXL = 400_000


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    with open(sys.argv[1]) as f:
        header = f.read().split()
    if len(header) != 3:
        fail("bad instance header")
    n, q, w = int(header[0]), int(header[1]), int(header[2])

    try:
        raw = open(sys.argv[2]).read()
    except Exception:
        fail("cannot read participant output")

    toks = raw.split()
    pos = [0]

    def nxt():
        if pos[0] >= len(toks):
            fail("unexpected EOF in output")
        t = toks[pos[0]]
        pos[0] += 1
        if len(t) > MAXTOK:
            fail("token too long")
        return t

    def nxt_int():
        t = nxt()
        try:
            return int(t)
        except Exception:
            fail("non-integer token (nan/inf/garbage) where an integer was expected")

    R = nxt_int()
    L = nxt_int()
    if L < 0 or L > MAXL:
        fail("bad instruction count L")
    if R != n + L:
        fail("R must equal n + L")

    ops = []
    mult_count = 0
    add_count = 0
    for i in range(L):
        kind = nxt()
        if kind not in ("A", "S", "M"):
            fail("bad opcode %r" % kind)
        a = nxt_int()
        b = nxt_int()
        cur_reg = n + i
        if not (0 <= a < cur_reg):
            fail("operand out of causal range at instruction %d" % i)
        if kind in ("A", "S"):
            if not (0 <= b < cur_reg):
                fail("operand out of causal range at instruction %d" % i)
            add_count += 1
        ops.append((kind, a, b))
        if kind == "M":
            c = b % q
            if c not in (0, 1, q - 1):
                mult_count += 1

    marker = nxt()
    if marker != "O":
        fail("missing 'O' output marker")
    outs = []
    for _ in range(n):
        o = nxt_int()
        if not (0 <= o < n + L):
            fail("output register out of range")
        outs.append(o)

    if pos[0] != len(toks):
        fail("trailing garbage after expected output")

    # ---- feasibility: apply the DAG to every standard basis vector ----
    for j in range(n):
        reg = [0] * (n + L)
        reg[j] = 1
        for i, (kind, a, b) in enumerate(ops):
            idx = n + i
            if kind == "A":
                reg[idx] = (reg[a] + reg[b]) % q
            elif kind == "S":
                reg[idx] = (reg[a] - reg[b]) % q
            else:
                reg[idx] = (reg[a] * (b % q)) % q
        for k in range(n):
            expected = pow(w, (j * k) % n, q)
            if reg[outs[k]] != expected:
                fail("mismatch at input e_%d, output index %d (got %d want %d)"
                     % (j, k, reg[outs[k]], expected))

    # ---- internal baseline: naive full n x n NTT matrix, one mult per nontrivial cell ----
    B = 0
    for j in range(n):
        for k in range(n):
            v = pow(w, (j * k) % n, q)
            if v not in (0, 1, q - 1):
                B += 1
    if B <= 0:
        fail("degenerate baseline")

    F = mult_count + ADD_COST * add_count
    ratio = min(1.0, SCALE * B / max(1e-9, F))
    print("n=%d B=%d mults=%d adds=%d F=%.2f Ratio: %.6f" % (n, B, mult_count, add_count, F, ratio))


if __name__ == "__main__":
    main()
