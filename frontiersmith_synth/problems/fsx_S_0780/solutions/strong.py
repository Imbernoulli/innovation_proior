# TIER: strong
# The insight: mine repeated ABSTRACT shapes, not literal text.
#   1. operand-canonicalization -- for each commutative (ADD/MUL) instruction,
#      sort its two operands by a kind-based canonical key (internal-temp <
#      external-parameter < constant, ties broken by relative position / value)
#      so occurrences that differ only by written operand order collapse to the
#      same signature.
#   2. parameterized-abstraction-mining -- any operand that is NOT an internal
#      temp of the window and NOT a baked constant (i.e. a global input var, or
#      a caller-local value produced before the window) is abstracted into a
#      formal macro parameter; repeated use of the SAME actual value inside one
#      occurrence is detected by raw-token identity and mapped to the SAME
#      parameter, preserving the abstraction's true arity.
#   3. library-amortization -- candidate shapes are ranked by (L-1)*count-3*L
#      (definition paid once, call paid per use) and the profitable ones,
#      largest-saving first and non-overlapping, are realized as a library of
#      up to 8 macros.
import sys

WINLENS = [2, 3, 4, 5]
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


def classify(tok, s):
    if tok[0] == "t":
        idx = int(tok[1:])
        if idx >= s:
            return ("T", idx - s)
        return ("EXTP", tok)
    if tok[0] == "c":
        return ("C", int(tok[1:]))
    # x<idx>
    return ("EXTP", tok)


def sort_key(kind):
    if kind[0] == "T":
        return (0, kind[1])
    if kind[0] == "EXTP":
        return (1, 0)
    return (2, kind[1])


def canonical_signature(instr, s, L):
    rows = []
    for r in range(L):
        op, a, b = instr[s + r]
        ka = classify(a, s)
        kb = classify(b, s)
        if op in ("ADD", "MUL"):
            pair = tuple(sorted((ka, kb), key=sort_key))
        else:
            pair = (ka, kb)
        rows.append((op, pair[0], pair[1]))

    paramid = {}
    order = []
    sig = []
    for (op, ka, kb) in rows:
        fk = []
        for k in (ka, kb):
            if k[0] == "EXTP":
                raw = k[1]
                if raw not in paramid:
                    paramid[raw] = len(paramid)
                    order.append(raw)
                fk.append(("P", paramid[raw]))
            else:
                fk.append(k)
        sig.append((op, fk[0], fk[1]))
    return tuple(sig), order  # order[i] = raw token bound to param i, for THIS occurrence


def compress_program(instr, out_idx, Lp, claimed):
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
            new_instr.append(("CALL", name, remapped_args))
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

    groups = {}  # sig -> list of (prog_idx, s, L, args[])
    for pi, (instr, out_idx) in enumerate(progs):
        Lp = len(instr)
        for L in WINLENS:
            for s in range(0, Lp - L + 1):
                if not valid_window(instr, Lp, out_idx, s, L):
                    continue
                sig, order = canonical_signature(instr, s, L)
                groups.setdefault(sig, []).append((pi, s, L, order))

    cand = []
    for sig, occ in groups.items():
        L = len(sig)
        count = len(occ)
        save = (L - 1) * count - 3 * L
        if save > 0:
            cand.append((save, sig, occ))
    cand.sort(key=lambda t: -t[0])

    claimed_ranges = {}
    macros = []
    per_prog_claims = {}

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
        for (pi, s, _L, order) in kept:
            claimed_ranges.setdefault(pi, []).append((s, L))
            per_prog_claims.setdefault(pi, []).append((s, L, name, order))

    out = ["MACROS %d" % len(macros)]
    for (name, sig) in macros:
        L = len(sig)
        arity = 0
        for (op, ka, kb) in sig:
            for k in (ka, kb):
                if k[0] == "P":
                    arity = max(arity, k[1] + 1)
        out.append("MACRO %s %d %d" % (name, arity, L))
        for r in range(L):
            op, ka, kb = sig[r]
            def tokstr(k):
                if k[0] == "T":
                    return "t%d" % k[1]
                if k[0] == "P":
                    return "p%d" % k[1]
                return "c%d" % k[1]
            out.append("%s %s %s" % (op, tokstr(ka), tokstr(kb)))
        out.append("RET t%d" % (L - 1))

    out.append("PROGRAMS %d" % N)
    for pi, (instr, out_idx) in enumerate(progs):
        Lp = len(instr)
        claims = sorted(per_prog_claims.get(pi, []), key=lambda t: t[0])
        new_instr, new_out = compress_program(instr, out_idx, Lp, claims)
        out.append("PROGRAM %d %d" % (pi, len(new_instr)))
        for ins in new_instr:
            if ins[0] == "CALL":
                _, name, args = ins
                out.append("CALL %s %s" % (name, " ".join(args)) if args else "CALL %s" % name)
            else:
                op, a, b = ins
                out.append("%s %s %s" % (op, a, b))
        out.append("OUT t%d" % new_out)

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
