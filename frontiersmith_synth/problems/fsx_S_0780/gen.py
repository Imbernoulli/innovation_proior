import sys, random

# gen.py <testId> -- prints ONE macro-library-compression corpus to stdout.
#
# Each program in the corpus is a flat straight-line program (SLP) over M shared
# input variables x0..x{M-1}, built by concatenating "slots". Each slot is one of:
#   - a TEMPLATE instantiation: one of 4 hidden abstract SLPs (fixed op-shape,
#     fixed constants, formal parameters) bound to randomly chosen actual operands
#     each time, with commutative (ADD/MUL) instructions independently randomly
#     operand-swapped at each occurrence. The literal text therefore differs on
#     every occurrence even though the abstract shape recurs dozens of times.
#   - a BOILERPLATE block: a fixed 4-op sequence using literally identical
#     operands (x0,x1) every single time -- a genuine, easy, literal duplicate
#     (this is what a naive frequent-substring miner CAN find).
#   - a FILLER op: one random single instruction (never profitable to macro-ize
#     under the amortized cost model, regardless of accidental repeats).
#
# The corpus is emitted FLAT (no macros, no CALLs) -- the solver must discover
# and re-introduce structure itself.

M = 6           # shared input variables x0..x{M-1} in every program
P = 1000000007  # modulus documented for the solver; equivalence is checked mod P
COMMUTATIVE = {"ADD", "MUL"}

# Each template: dict(arity=int, body=[(op, opA, opB), ...]) with operand kinds
#   ("P", idx)  formal parameter idx
#   ("T", idx)  reference to this template's own body instruction idx (idx < position)
#   ("C", val)  a fixed constant baked into the template shape
# Design rule: a COMMUTATIVE instruction never combines two ("P", *) operands
# directly (that pairing has no canonical order without first knowing parameter
# identity); SUB may combine anything since its operand order is never permuted.
TEMPLATES = [
    dict(arity=2, body=[
        ("MUL", ("P", 0), ("C", 3)),
        ("SUB", ("P", 1), ("T", 0)),
        ("ADD", ("T", 1), ("P", 0)),
    ]),
    dict(arity=3, body=[
        ("SUB", ("P", 0), ("P", 1)),
        ("MUL", ("T", 0), ("P", 2)),
        ("ADD", ("P", 2), ("C", 2)),
        ("ADD", ("T", 1), ("T", 2)),
    ]),
    dict(arity=2, body=[
        ("MUL", ("P", 0), ("C", 5)),
        ("MUL", ("T", 0), ("P", 1)),
        ("SUB", ("P", 1), ("T", 1)),
    ]),
    dict(arity=3, body=[
        ("SUB", ("P", 0), ("P", 1)),
        ("MUL", ("T", 0), ("P", 2)),
        ("ADD", ("P", 2), ("C", 7)),
        ("MUL", ("T", 1), ("T", 2)),
    ]),
]

# Fixed literal boilerplate: identical text every occurrence, never swapped.
BOILERPLATE = [
    ("ADD", ("X", 0), ("X", 1)),
    ("MUL", ("T", 0), ("X", 0)),
    ("SUB", ("T", 1), ("X", 1)),
    ("ADD", ("T", 2), ("X", 0)),
    ("MUL", ("T", 3), ("X", 1)),
]


def fmt_operand(kind, params, temp_base):
    tag, val = kind
    if tag == "P":
        return params[val]
    if tag == "C":
        return "c%d" % val
    if tag == "T":
        return "t%d" % (temp_base + val)
    if tag == "X":
        return "x%d" % val
    raise ValueError(kind)


def emit_block(body, params, temp_base, rng, allow_swap):
    lines = []
    for (op, a, b) in body:
        ta, tb = a, b
        if allow_swap and op in COMMUTATIVE and rng.random() < 0.5:
            ta, tb = b, a
        sa = fmt_operand(ta, params, temp_base)
        sb = fmt_operand(tb, params, temp_base)
        lines.append("%s %s %s" % (op, sa, sb))
    return lines


def gen_program(rng):
    instr = []          # flat list of "OP a b" strings; position == temp index
    slot_outputs = []   # global temp indices of each completed slot's final value

    def pool_pick():
        if slot_outputs and rng.random() < 0.35:
            return "t%d" % rng.choice(slot_outputs)
        return "x%d" % rng.randrange(M)

    def const_pick():
        return "c%d" % rng.randint(-9, 9)

    S = rng.randint(6, 10)
    for _ in range(S):
        base = len(instr)
        r = rng.random()
        if r < 0.50:
            tpl = TEMPLATES[rng.randrange(len(TEMPLATES))]
            params = [pool_pick() for _ in range(tpl["arity"])]
            block = emit_block(tpl["body"], params, base, rng, allow_swap=True)
        elif r < 0.85:
            block = emit_block(BOILERPLATE, [], base, rng, allow_swap=False)
        else:
            op = rng.choice(["ADD", "SUB", "MUL"])
            a_tok = pool_pick() if rng.random() < 0.8 else const_pick()
            b_tok = pool_pick() if rng.random() < 0.8 else const_pick()
            block = ["%s %s %s" % (op, a_tok, b_tok)]
        instr.extend(block)
        slot_outputs.append(len(instr) - 1)

    out_idx = slot_outputs[-1]
    return instr, out_idx


# testId -> corpus size (difficulty ladder: small -> large)
def n_for(tid):
    return 20 + 6 * tid


def main():
    tid = int(sys.argv[1])
    if not (1 <= tid <= 10):
        raise SystemExit("testId out of range")
    N = n_for(tid)
    seed = 700000 + 97 * tid
    rng = random.Random(seed)

    lines = ["%d %d %d %d" % (N, M, P, tid)]
    for k in range(N):
        instr, out_idx = gen_program(rng)
        lines.append("PROGRAM %d %d" % (k, len(instr)))
        lines.extend(instr)
        lines.append("OUT t%d" % out_idx)
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
