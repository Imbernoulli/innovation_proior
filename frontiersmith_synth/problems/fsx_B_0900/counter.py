import sys

# counter.py -- Format D checker for "segmented in-place permute" (warehouse pallets
# across pricey doorways).
#
#   <in>:  n m K D
#          seg[0..n-1]     -- segment id (0..m-1) of every slot
#          perm[0..n-1]    -- perm[i] = pallet currently AT slot i; pallet i belongs at
#                             slot i (the program must sort the array back to identity)
#   <out>: T
#          T operation lines, each  "M a b"  or  "S a b"
#            M a b : MOVE.  b must currently be empty, a must currently be occupied.
#                    After: b holds a's old content, a becomes empty.
#            S a b : SWAP.  a and b must both currently be occupied; their contents
#                    are exchanged.
#          a, b are each either an integer slot index 0<=idx<n, or a register token
#          "R<k>" with 0<=k<K.  There are K scratch registers, each holding at most one
#          pallet, all EMPTY at the start and required EMPTY again at the end.
#
# Cost rule for every op: if BOTH endpoints are real slots, cost = 1 when they share a
# segment, D when they don't (crossing a doorway); if EITHER endpoint is a register, cost
# = 1 flat (a register is a small depot reachable cheaply from any aisle).
#
# 1) Parse & validate the op list strictly (bounds, well-formedness, occupancy rules,
#    finiteness) -> any violation prints Ratio: 0.0 and exits 0.
# 2) Simulate every op; the final slot contents must equal identity (slot i holds pallet
#    i for all i) AND every register must be empty -> else Ratio: 0.0.
# 3) Objective (minimize) = total cost F.
#    Baseline B: the checker's OWN structure-blind reference -- decompose perm into
#    cycles (leftmost-index start, following perm forward) and realize each cycle as the
#    textbook star-of-transpositions from the cycle's smallest-index element, but costed
#    as if built with MOVE only (no SWAP primitive): every transposition goes through a
#    shared temp bay (evacuate-a, move-b-into-a, move-temp-into-b), i.e. cost =
#    edgecost(anchor,b) + 2 per transposition instead of a bare SWAP's edgecost(anchor,b).
#    Ratio = min(1, 0.1 * B / F).

MAX_OPS_PER_SLOT = 8


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def decompose_cycles(perm, n):
    seen = [False] * n
    cycles = []
    for i in range(n):
        if seen[i]:
            continue
        if perm[i] == i:
            seen[i] = True
            continue
        cyc = []
        j = i
        while not seen[j]:
            seen[j] = True
            cyc.append(j)
            j = perm[j]
        cycles.append(cyc)
    return cycles


def edgecost(a, b, seg, D):
    return 1 if seg[a] == seg[b] else D


def naive_detour_baseline(perm, n, seg, D):
    """The checker's own construction cost: star-of-transpositions from cyc[0], each
    transposition realized via a 3-move temp-bay detour (cost edgecost+2) instead of a
    single SWAP."""
    total = 0
    for cyc in decompose_cycles(perm, n):
        L = len(cyc)
        anchor = cyc[0]
        for k in range(1, L):
            b = cyc[k]
            total += edgecost(anchor, b, seg, D) + 2
    return total


def parse_token(tok, n, K):
    if len(tok) > 0 and (tok[0] == 'R' or tok[0] == 'r'):
        rest = tok[1:]
        if rest == "" or not rest.isdigit():
            raise ValueError("bad register token %r" % tok)
        idx = int(rest)
        if not (0 <= idx < K):
            raise ValueError("register index out of range %r" % tok)
        return ('R', idx)
    # strict integer (rejects nan/inf/floats/garbage)
    if not (tok.lstrip('-').isdigit()):
        raise ValueError("bad slot token %r" % tok)
    v = int(tok)
    if not (0 <= v < n):
        raise ValueError("slot index out of range %r" % tok)
    return ('S', v)


def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    it = iter(inp)
    try:
        n = int(next(it)); m = int(next(it)); K = int(next(it)); D = int(next(it))
    except Exception:
        fail("bad header")
    if not (2 <= n <= 200000):
        fail("bad n")
    if not (1 <= m <= n):
        fail("bad m")
    if not (0 <= K <= 16):
        fail("bad K")
    if not (1 <= D <= 1000):
        fail("bad D")
    try:
        seg = [int(next(it)) for _ in range(n)]
        perm = [int(next(it)) for _ in range(n)]
    except Exception:
        fail("bad seg/perm arrays -- generator bug")
    if any(not (0 <= s < m) for s in seg):
        fail("segment id out of range -- generator bug")
    if sorted(perm) != list(range(n)):
        fail("perm is not a permutation -- generator bug")

    # ---- parse participant output ----
    if not out:
        fail("empty output")
    oit = iter(out)
    try:
        Ttok = next(oit)
        if not Ttok.lstrip('-').isdigit():
            fail("bad T")
        T = int(Ttok)
    except StopIteration:
        fail("empty output")
    if T < 0:
        fail("T < 0")
    max_ops = MAX_OPS_PER_SLOT * n + 64
    if T > max_ops:
        fail("T too large (> %d)" % max_ops)

    ops = []
    try:
        for _ in range(T):
            opc = next(oit)
            if opc not in ("M", "S"):
                fail("bad opcode %r" % opc)
            a_tok = next(oit)
            b_tok = next(oit)
            a = parse_token(a_tok, n, K)
            b = parse_token(b_tok, n, K)
            if a == b:
                fail("op references the same location twice")
            ops.append((opc, a, b))
    except SystemExit:
        raise
    except StopIteration:
        fail("truncated output")
    except ValueError as e:
        fail(str(e))
    except Exception:
        fail("malformed op token")

    # every declared token must be one of the T ops -- no trailing garbage allowed
    leftover = list(oit)
    if leftover:
        fail("trailing garbage after the declared %d operations (%d extra token(s))" % (T, len(leftover)))

    # ---- simulate ----
    slot_content = list(perm)          # slot_content[i] = pallet currently at slot i
    slot_occ = [True] * n               # every slot starts occupied
    reg_content = [None] * K
    reg_occ = [False] * K

    def get_occ(loc):
        kind, idx = loc
        return slot_occ[idx] if kind == 'S' else reg_occ[idx]

    def get_val(loc):
        kind, idx = loc
        return slot_content[idx] if kind == 'S' else reg_content[idx]

    def set_val(loc, val, occ):
        kind, idx = loc
        if kind == 'S':
            slot_content[idx] = val
            slot_occ[idx] = occ
        else:
            reg_content[idx] = val
            reg_occ[idx] = occ

    F = 0
    for step, (opc, a, b) in enumerate(ops):
        if opc == "M":
            if not get_occ(a):
                fail("MOVE from empty source at step %d" % step)
            if get_occ(b):
                fail("MOVE onto occupied destination at step %d" % step)
            val = get_val(a)
            set_val(a, None, False)
            set_val(b, val, True)
        else:  # SWAP
            if not get_occ(a) or not get_occ(b):
                fail("SWAP with an empty endpoint at step %d" % step)
            va, vb = get_val(a), get_val(b)
            set_val(a, vb, True)
            set_val(b, va, True)
        if a[0] == 'S' and b[0] == 'S':
            F += edgecost(a[1], b[1], seg, D)
        else:
            F += 1

    if slot_content != list(range(n)):
        bad = next(i for i in range(n) if slot_content[i] != i)
        fail("final arrangement wrong (first mismatch at slot %d)" % bad)
    if any(reg_occ):
        fail("a scratch register is left occupied at the end")

    if F <= 0:
        fail("zero-cost program cannot be correct for a non-identity permutation")

    B = naive_detour_baseline(perm, n, seg, D)
    if B <= 0:
        fail("degenerate instance: identity permutation -- generator bug")

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("n=%d T=%d F=%d B=%d Ratio: %.6f" % (n, T, F, B, ratio))


if __name__ == "__main__":
    main()
