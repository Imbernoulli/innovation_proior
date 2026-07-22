import sys

# Format D checker -- cheapest plotter program for a D4-symmetric emblem.
#   1) Parse N and the anchor list from <in>.
#   2) Interpret the participant's flat token stream (<out>) as a straight-line
#      DOT/DEF/CALL/END program (see statement.md for the exact grammar).
#   3) Feasibility: the inked set S must contain every anchor and be invariant
#      under all 8 D4 symmetries about the grid center.
#   4) Objective (minimize) = total DOT+CALL instruction count R.
#      Baseline B = |orbit closure of the anchors| (the fully-unrolled, macro-free
#      program length). Ratio = min(1, 0.1 * B / R).

MAXR = 4000
MAXMACROS = 1000


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def apply_t(t, x, y, N):
    if t == 0: return (x, y)
    if t == 1: return (N - 1 - y, x)
    if t == 2: return (N - 1 - x, N - 1 - y)
    if t == 3: return (y, N - 1 - x)
    if t == 4: return (N - 1 - x, y)
    if t == 5: return (x, N - 1 - y)
    if t == 6: return (y, x)
    if t == 7: return (N - 1 - y, N - 1 - x)
    return None


def main():
    inp = open(sys.argv[1]).read().split()
    it_in = iter(inp)
    try:
        N = int(next(it_in))
        A = int(next(it_in))
        if not (4 <= N <= 64 and N % 2 == 0 and 1 <= A <= 400):
            fail("bad header")
        anchors = []
        for _ in range(A):
            x = int(next(it_in)); y = int(next(it_in))
            if not (0 <= x < N and 0 <= y < N):
                fail("bad anchor in input")
            anchors.append((x, y))
    except (StopIteration, ValueError):
        fail("malformed input file")

    # ---- parse participant program (flat whitespace token stream) ----
    out_toks = open(sys.argv[2]).read().split()
    it = iter(out_toks)

    def next_int(what):
        try:
            tok = next(it)
        except StopIteration:
            fail("truncated stream while reading %s" % what)
        try:
            return int(tok)
        except ValueError:
            fail("non-integer token %r while reading %s" % (tok, what))

    macros = []      # list of frozenset((x,y))
    S = set()
    instr = 0
    in_def = False
    cur_local = None

    while True:
        tok = next(it, None)
        if tok is None:
            break
        TK = tok.upper()
        if TK == "DOT":
            x = next_int("DOT x"); y = next_int("DOT y")
            if not (0 <= x < N and 0 <= y < N):
                fail("DOT (%d,%d) out of range" % (x, y))
            (cur_local if in_def else S).add((x, y))
            instr += 1
        elif TK == "CALL":
            k = next_int("CALL macro id"); g = next_int("CALL transform id")
            if not (0 <= k < len(macros)):
                fail("CALL references undefined/forward macro %d" % k)
            if not (0 <= g <= 7):
                fail("CALL bad transform id %d" % g)
            transformed = set(apply_t(g, px, py, N) for (px, py) in macros[k])
            if in_def:
                cur_local |= transformed
            else:
                S |= transformed
            instr += 1
        elif TK == "DEF":
            if in_def:
                fail("nested DEF is not allowed")
            k = next_int("DEF id")
            if k != len(macros):
                fail("DEF %d out of sequence (macros must be numbered 0,1,2,... in order)" % k)
            in_def = True
            cur_local = set()
        elif TK == "END":
            if not in_def:
                fail("END without a matching DEF")
            macros.append(frozenset(cur_local))
            in_def = False
            cur_local = None
            if len(macros) > MAXMACROS:
                fail("too many macros (> %d)" % MAXMACROS)
        else:
            fail("unknown instruction token %r" % tok)

        if instr > MAXR:
            fail("instruction budget exceeded (> %d)" % MAXR)

    if in_def:
        fail("unterminated DEF (missing END)")
    if instr == 0:
        fail("empty program")

    # ---- feasibility: coverage ----
    for a in anchors:
        if a not in S:
            fail("anchor %r not covered" % (a,))

    # ---- feasibility: invariance under all 8 symmetries ----
    for t in range(1, 8):
        for (x, y) in S:
            if apply_t(t, x, y, N) not in S:
                fail("inked set not invariant under symmetry %d" % t)

    # ---- baseline B = orbit closure of the anchors ----
    closure = set()
    for (x, y) in anchors:
        for t in range(8):
            closure.add(apply_t(t, x, y, N))
    B = len(closure)
    if B == 0:
        fail("degenerate empty closure")

    R = instr
    ratio = min(1.0, 0.1 * B / R)
    print("R=%d B=%d |S|=%d Ratio: %.6f" % (R, B, len(S), ratio))


if __name__ == "__main__":
    main()
