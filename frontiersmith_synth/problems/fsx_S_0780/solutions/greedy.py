# TIER: greedy
# The "obvious" approach: mine EXACT, byte-identical contiguous instruction
# blocks (only renaming purely-relative internal temp positions -- otherwise no
# two blocks would ever line up at all) and turn repeated ones into 0-argument
# macros. No operand-canonicalization (commutative swaps are NOT normalized) and
# NO parameterization (a block that references a different actual variable/const,
# or a caller-local value from before the block, each time is simply never
# matched). This finds the literal boilerplate duplicate but is blind to the
# parameterized abstract templates that dominate the corpus.
import sys

MAXLEN = [2, 3, 4, 5]
MAX_MACROS = 8


def read_instance():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); P = int(next(it)); SEED = int(next(it))
    progs = []
    for k in range(N):
        assert next(it) == "PROGRAM"
        kk = int(next(it)); L = int(next(it))
        instr = []
        for _ in range(L):
            op = next(it); a = next(it); b = next(it)
            instr.append((op, a, b))
        assert next(it) == "OUT"
        out_tok = next(it)
        out_idx = int(out_tok[1:])
        progs.append((instr, out_idx))
    return N, M, P, progs


def valid_window(instr, Lp, out_idx, s, L):
    # no reference from outside the window to a non-final internal temp
    last = s + L - 1
    for q in range(s + L, Lp):
        _, a, b = instr[q]
        for tok in (a, b):
            if tok[0] == "t":
                idx = int(tok[1:])
                if s <= idx < last:
                    return False
    if s <= out_idx < last:
        return False
    return True


def literal_signature(instr, s, L):
    sig = []
    args_present = False
    for r in range(L):
        op, a, b = instr[s + r]
        enc = []
        for tok in (a, b):
            if tok[0] == "t":
                idx = int(tok[1:])
                if idx >= s:
                    enc.append(("T", idx - s))
                else:
                    return None  # external temp ref -> not literal-eligible
            else:
                enc.append(("LIT", tok))
        sig.append((op, enc[0], enc[1]))
    return tuple(sig)


def compress_program(instr, out_idx, Lp, claimed):
    # claimed: sorted list of (s, L, macroname, args[])  non-overlapping
    old_to_new = [-1] * Lp
    new_instr = []
    i = 0
    ci = 0
    while i < Lp:
        if ci < len(claimed) and claimed[ci][0] == i:
            s, L, name, args = claimed[ci]
            remapped_args = []
            for tok in args:
                if tok[0] == "t":
                    remapped_args.append("t%d" % old_to_new[int(tok[1:])])
                else:
                    remapped_args.append(tok)
            if remapped_args:
                new_instr.append(("CALL", name, remapped_args))
            else:
                new_instr.append(("CALL0", name, []))
            old_to_new[s + L - 1] = len(new_instr) - 1
            i = s + L
            ci += 1
        else:
            op, a, b = instr[i]
            na = "t%d" % old_to_new[int(a[1:])] if a[0] == "t" else a
            nb = "t%d" % old_to_new[int(b[1:])] if b[0] == "t" else b
            new_instr.append((op, na, nb))
            old_to_new[i] = len(new_instr) - 1
            i += 1
    new_out = old_to_new[out_idx]
    return new_instr, new_out


def main():
    N, M, P, progs = read_instance()

    groups = {}  # sig -> list of (prog_idx, s, L, args=[])
    for pi, (instr, out_idx) in enumerate(progs):
        Lp = len(instr)
        for L in MAXLEN:
            for s in range(0, Lp - L + 1):
                if not valid_window(instr, Lp, out_idx, s, L):
                    continue
                sig = literal_signature(instr, s, L)
                if sig is None:
                    continue
                groups.setdefault(sig, []).append((pi, s, L))

    cand = []
    for sig, occ in groups.items():
        L = len(sig)
        count = len(occ)
        save = (L - 1) * count - 3 * L
        if save > 0:
            cand.append((save, sig, occ))
    cand.sort(key=lambda t: -t[0])

    claimed_ranges = {}  # prog_idx -> list of (s,L)
    macros = []          # (name, sig)
    per_prog_claims = {} # prog_idx -> list of (s,L,name,args)

    def overlaps(pi, s, L):
        for (cs, cl) in claimed_ranges.get(pi, []):
            if not (s + L <= cs or cs + cl <= s):
                return True
        return False

    for save, sig, occ in cand:
        if len(macros) >= MAX_MACROS:
            break
        L = len(sig)
        kept = [o for o in occ if not overlaps(o[0], o[1], L)]
        count = len(kept)
        if count == 0:
            continue
        new_save = (L - 1) * count - 3 * L
        if new_save <= 0:
            continue
        name = "m%d" % len(macros)
        macros.append((name, sig))
        for (pi, s, _L) in kept:
            claimed_ranges.setdefault(pi, []).append((s, L))
            per_prog_claims.setdefault(pi, []).append((s, L, name, []))

    out = ["MACROS %d" % len(macros)]
    for (name, sig) in macros:
        L = len(sig)
        out.append("MACRO %s 0 %d" % (name, L))
        for r in range(L):
            op, encA, encB = sig[r]
            def tokstr(enc):
                if enc[0] == "T":
                    return "t%d" % enc[1]
                return enc[1]  # literal token (x.. or c..)
            out.append("%s %s %s" % (op, tokstr(encA), tokstr(encB)))
        out.append("RET t%d" % (L - 1))

    out.append("PROGRAMS %d" % N)
    for pi, (instr, out_idx) in enumerate(progs):
        Lp = len(instr)
        claims = sorted(per_prog_claims.get(pi, []), key=lambda t: t[0])
        new_instr, new_out = compress_program(instr, out_idx, Lp, claims)
        out.append("PROGRAM %d %d" % (pi, len(new_instr)))
        for ins in new_instr:
            if ins[0] in ("CALL", "CALL0"):
                _, name, args = ins
                out.append("CALL %s %s" % (name, " ".join(args)))
            else:
                op, a, b = ins
                out.append("%s %s %s" % (op, a, b))
        out.append("OUT t%d" % new_out)

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
