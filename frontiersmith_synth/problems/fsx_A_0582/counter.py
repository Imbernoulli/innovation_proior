import sys, math, hashlib, random

# Format D checker -- minimal-operation straight-line program recompression.
#   1) Parse the GIVEN ritual program G (over F_p, inputs x0..x7) from <in>.
#   2) Parse the participant's replacement program S from <out>  (strict schema).
#   3) EQUIVALENCE GATE: S must agree with G at K deterministic evaluation points
#      over F_p (S computes the same function).  Any mismatch -> Ratio 0.0.
#   4) Objective (MINIMIZE) = number of arithmetic ops (add/sub/mul; const is free).
#      Baseline B = op count of G (what "echo the ritual" achieves).
#      Score curve keeps headroom above the reference recurrence:
#         ratio = clamp_[0, 0.98]( 0.10 + 0.280 * ln(B / F) )
#      -> echo (F=B) scores 0.10; every 2x reduction adds ~0.19; near-optimal
#         recurrences land ~0.4..0.9 with the ceiling still open above.

P = 2147483647
NIN = 8
S_COEF = 0.280
CAP = 0.98
K_POINTS = 256
MAXL = 200000           # hard cap on submitted program length


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def parse_program(tokens, it, L, tag):
    """Read L instructions from token iterator `it`. Returns list of instrs.
    Value indices: 0..7 inputs; instruction t -> index 8+t. References must be
    strictly earlier and non-negative."""
    prog = []
    for t in range(L):
        try:
            op = next(it)
        except StopIteration:
            fail("%s: truncated at instruction %d" % (tag, t))
        cur = NIN + t                       # index this instruction will occupy
        if op == "const":
            try:
                v = int(next(it))
            except (StopIteration, ValueError):
                fail("%s: bad const operand" % tag)
            if v < 0 or v >= P:
                fail("%s: const out of range" % tag)
            prog.append(("const", v % P, None))
        elif op in ("add", "sub", "mul"):
            try:
                i = int(next(it)); j = int(next(it))
            except (StopIteration, ValueError):
                fail("%s: bad %s operand" % (tag, op))
            if not (0 <= i < cur) or not (0 <= j < cur):
                fail("%s: index out of range in %s" % (tag, op))
            prog.append((op, i, j))
        else:
            fail("%s: unknown op '%s'" % (tag, op[:16]))
    return prog


def evaluate(prog, xs):
    vals = list(xs)
    for ins in prog:
        op = ins[0]
        if op == "const":
            vals.append(ins[1])
        elif op == "add":
            vals.append((vals[ins[1]] + vals[ins[2]]) % P)
        elif op == "sub":
            vals.append((vals[ins[1]] - vals[ins[2]]) % P)
        else:
            vals.append((vals[ins[1]] * vals[ins[2]]) % P)
    return vals[-1]


def opcount(prog):
    return sum(1 for x in prog if x[0] in ("add", "sub", "mul"))


def main():
    in_text = open(sys.argv[1]).read()
    out_text = open(sys.argv[2]).read()

    # ---- parse the given ritual G ----
    itin = iter(in_text.split())
    try:
        p = int(next(itin)); L = int(next(itin))
    except (StopIteration, ValueError):
        fail("bad instance header")
    if p != P:
        fail("wrong field")
    G = parse_program(in_text, itin, L, "given")

    # ---- parse the submission S ----
    otoks = out_text.split()
    if not otoks:
        fail("empty output")
    ito = iter(otoks)
    try:
        Ls = int(next(ito))
    except (StopIteration, ValueError):
        fail("bad submission length")
    if Ls < 1:
        fail("need >= 1 instruction")
    if Ls > MAXL:
        fail("submission too long")
    S = parse_program(out_text, ito, Ls, "sub")
    # reject trailing garbage (strict token count)
    if next(ito, None) is not None:
        fail("trailing tokens in submission")

    # ---- equivalence gate: agree with G on K deterministic points over F_p ----
    seed = int.from_bytes(hashlib.sha256(in_text.encode()).digest()[:8], "big") ^ 0x5EED1234
    rng = random.Random(seed)
    for _ in range(K_POINTS):
        xs = [rng.randrange(0, P) for _ in range(NIN)]
        if evaluate(G, xs) != evaluate(S, xs):
            fail("not equivalent to the given program")

    B = opcount(G)
    F = opcount(S)
    if F < 1:
        fail("no arithmetic ops")
    ratio = 0.10 + S_COEF * math.log(B / F)
    ratio = max(0.0, min(CAP, ratio))
    print("B=%d F=%d Ratio: %.6f" % (B, F, ratio))


if __name__ == "__main__":
    main()
