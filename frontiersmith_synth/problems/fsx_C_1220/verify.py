#!/usr/bin/env python3
"""verify.py <in> <out> <ans> -- scores a submitted type SCHEME for the
let-generalization inference task.

Artifact (participant stdout), strict schema:
    k
    k lines, one per definition-local type variable i=0..k-1, in order:
        FIX <closed type tokens>      -- variable is bound to this exact ground type
        GEN <label>                   -- variable is universally quantified under <label>

Scoring:
  1) SOUNDNESS (feasibility gate): substituting the submission into every
     DEFINITION constraint (FIX splices the ground type; GEN splices an opaque
     labeled leaf) must make both sides of each equation syntactically identical
     (ground constructors match; GEN leaves match iff same label). ANY violation
     -> Ratio 0.0.
  2) GENERALITY (objective): for each held-out USE site, a FRESH copy of every
     GEN label is instantiated (same label = same fresh slot only WITHIN that one
     use; a different use gets independent fresh slots -- this is the "instantiate
     at the generalization point" semantics). The use PASSES iff all of its
     equations are then simultaneously consistent. F = #uses that pass.
  3) Baseline B = u/10 (a calibrated denominator: a solver that ignores every
     use site and blind-guesses one default ground type for every free slot
     typically clears only a small slice of the batch by chance, so that
     baseline is pinned near Ratio 0.1; a scheme that is both sound AND general
     clears most of the batch and scores close to, but never quite at, 1.0 --
     at least one use in every instance is planted so that NO scheme, however
     general, can satisfy it).
     Ratio = min(1000, 100*F/B) / 1000.
"""
import sys


class TokReader:
    def __init__(self, toks):
        self.toks = toks
        self.p = 0

    def eof(self):
        return self.p >= len(self.toks)

    def next(self):
        if self.p >= len(self.toks):
            raise ValueError("unexpected end of input")
        t = self.toks[self.p]
        self.p += 1
        return t

    def next_int(self):
        t = self.next()
        if not (t.lstrip("-").isdigit()):
            raise ValueError("expected integer, got %r" % t)
        return int(t)

    def parse_type(self, allow_var, depth=0):
        if depth > 12:
            raise ValueError("type nesting too deep")
        tok = self.next()
        if tok == "L":
            return ("L", self.parse_type(allow_var, depth + 1))
        if tok == "P":
            x = self.parse_type(allow_var, depth + 1)
            y = self.parse_type(allow_var, depth + 1)
            return ("P", x, y)
        if tok == "F":
            x = self.parse_type(allow_var, depth + 1)
            y = self.parse_type(allow_var, depth + 1)
            return ("F", x, y)
        if tok == "int":
            return ("int",)
        if tok == "bool":
            return ("bool",)
        if allow_var and len(tok) >= 2 and tok[0] == "t" and tok[1:].isdigit():
            return ("V", int(tok[1:]))
        raise ValueError("bad type token: %r" % tok)

    def expect(self, s):
        t = self.next()
        if t != s:
            raise ValueError("expected %r, got %r" % (s, t))


def fail(reason):
    print("INFEASIBLE: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    tr = TokReader(toks)
    k = tr.next_int()
    m = tr.next_int()
    u = tr.next_int()
    defs = []
    for _ in range(m):
        A = tr.parse_type(True)
        tr.expect("=")
        B = tr.parse_type(True)
        defs.append((A, B))
    uses = []
    for _ in range(u):
        tr.expect("USE")
        uid = tr.next_int()
        q = tr.next_int()
        eqs = []
        for _ in range(q):
            A = tr.parse_type(True)
            tr.expect("=")
            B = tr.parse_type(True)
            eqs.append((A, B))
        uses.append(eqs)
    return k, m, u, defs, uses


IDENT_OK = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")


def read_submission(path, k):
    with open(path) as f:
        toks = f.read().split()
    if not toks:
        fail("empty output")
    tr = TokReader(toks)
    try:
        kk = tr.next_int()
    except ValueError as e:
        fail(str(e))
    if kk != k:
        fail("header k=%d does not match instance k=%d" % (kk, k))
    sigma = {}
    for i in range(k):
        if tr.eof():
            fail("output truncated before variable t%d" % i)
        tag = tr.next()
        if tag == "FIX":
            try:
                t = tr.parse_type(False, 0)
            except ValueError as e:
                fail("t%d FIX: %s" % (i, e))
            sigma[i] = ("G", t)
        elif tag == "GEN":
            if tr.eof():
                fail("t%d GEN: missing label" % i)
            lbl = tr.next()
            if not lbl or any(c not in IDENT_OK for c in lbl) or lbl[0].isdigit():
                fail("t%d GEN: bad label %r" % (i, lbl))
            sigma[i] = ("L", lbl)
        else:
            fail("t%d: expected FIX or GEN, got %r" % (i, tag))
    if not tr.eof():
        fail("trailing tokens after %d variable lines" % k)
    return sigma


def is_finite_tree(t):
    if t[0] in ("int", "bool"):
        return True
    if t[0] == "L":
        return is_finite_tree(t[1])
    if t[0] in ("P", "F"):
        return is_finite_tree(t[1]) and is_finite_tree(t[2])
    return False


def apply_sigma(t, sigma):
    """Substitute every V-leaf via sigma. Ground FIX leaves splice their closed
    tree; GEN leaves become an opaque ('LBL', label) leaf."""
    if t[0] == "V":
        kind, val = sigma[t[1]]
        return val if kind == "G" else ("LBL", val)
    if t[0] in ("int", "bool"):
        return t
    if t[0] == "L":
        return ("L", apply_sigma(t[1], sigma))
    return (t[0], apply_sigma(t[1], sigma), apply_sigma(t[2], sigma))


def tree_eq(a, b):
    if a[0] != b[0]:
        return False
    if a[0] in ("int", "bool"):
        return True
    if a[0] == "LBL":
        return a[1] == b[1]
    if a[0] == "L":
        return tree_eq(a[1], b[1])
    return tree_eq(a[1], b[1]) and tree_eq(a[2], b[2])


def main():
    if len(sys.argv) < 3:
        print("Ratio: 0.0")
        return 0
    in_path, out_path = sys.argv[1], sys.argv[2]
    k, m, u, defs, uses = read_instance(in_path)
    sigma = read_submission(out_path, k)

    # every FIX target must be finite/closed (parser already forbids V and bounds
    # depth, so this also doubles as the nan/inf/garbage-token defense: only
    # int/bool/L/P/F tokens are ever accepted).
    for i, (kind, val) in sigma.items():
        if kind == "G" and not is_finite_tree(val):
            fail("t%d FIX target not a closed finite type" % i)

    # 1) soundness against every DEFINITION constraint
    for j, (A, B) in enumerate(defs):
        Ar = apply_sigma(A, sigma)
        Br = apply_sigma(B, sigma)
        if not tree_eq(Ar, Br):
            fail("submission violates definition constraint #%d" % j)

    if u == 0:
        # no held-out uses to grade (shouldn't happen with our generator, but be safe)
        print("Ratio: 0.100000")
        return 0

    # 2) generality against held-out uses, fresh instantiation PER use
    passed = 0
    for eqs in uses:
        local = {}   # label -> resolved ground tree, fresh for this use only
        ok = True
        for A, B in eqs:
            Ar = apply_sigma(A, sigma)
            Br = apply_sigma(B, sigma)
            # B-side of every use equation is always closed ground by construction;
            # resolve any LBL leaf on either side against this use's local binding.
            def resolve(t):
                if t[0] == "LBL":
                    if t[1] in local:
                        return local[t[1]]
                    return None  # unbound this use -> caller binds it
                return t

            rA = resolve(Ar)
            rB = resolve(Br)
            if rA is None and rB is None:
                # two never-before-seen GEN labels meeting each other: always
                # unifiable (bind both to a shared fresh marker)
                if Ar[1] not in local and Br[1] not in local:
                    marker = ("FRESH", Ar[1], Br[1])
                    local[Ar[1]] = marker
                    local[Br[1]] = marker
                continue
            if rA is None:
                local[Ar[1]] = Br if rB is None else rB
                continue
            if rB is None:
                local[Br[1]] = Ar if rA is None else rA
                continue
            if not tree_eq(rA, rB):
                ok = False
                break
        if ok:
            passed += 1

    F = passed
    B = u / 10.0
    sc = min(1000.0, 100.0 * F / max(1e-9, B))
    print("passed %d / %d held-out uses" % (F, u))
    print("Ratio: %.6f" % (sc / 1000.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
