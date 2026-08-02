# TIER: greedy
"""Eager, generation-order solver -- the obvious first attempt.

Solves the DEFINITION constraints correctly (same unifier as strong), but has
no notion of let-generalization: any variable the definitions leave free is
just committed, in FILE ORDER, to whatever ground type the FIRST use site that
touches it happens to want -- exactly as if the whole program (definition +
every call site) were solved as one flat monomorphic system rather than
generalizing at the let and re-instantiating fresh per use.

This is sound (every definition constraint is still satisfied) and scores fine
whenever a slot's uses all happen to agree on a type. It is quietly wrong the
moment two uses of the same polymorphic slot want different ground types: the
scheme it emits is a valid but overly-specific instance, not the principal one.
"""
import sys


class TokReader:
    def __init__(self, toks):
        self.toks = toks
        self.p = 0

    def next(self):
        t = self.toks[self.p]
        self.p += 1
        return t

    def next_int(self):
        return int(self.next())

    def parse_type(self):
        tok = self.next()
        if tok == "L":
            return ("L", self.parse_type())
        if tok == "P":
            return ("P", self.parse_type(), self.parse_type())
        if tok == "F":
            return ("F", self.parse_type(), self.parse_type())
        if tok == "int":
            return ("int",)
        if tok == "bool":
            return ("bool",)
        return ("V", int(tok[1:]))

    def expect(self, s):
        t = self.next()
        assert t == s


def read_instance():
    toks = sys.stdin.read().split()
    tr = TokReader(toks)
    k = tr.next_int()
    m = tr.next_int()
    u = tr.next_int()
    defs = []
    for _ in range(m):
        A = tr.parse_type()
        tr.expect("=")
        B = tr.parse_type()
        defs.append((A, B))
    uses = []
    for _ in range(u):
        tr.expect("USE")
        uid = tr.next_int()
        q = tr.next_int()
        eqs = []
        for _ in range(q):
            A = tr.parse_type()
            tr.expect("=")
            B = tr.parse_type()
            eqs.append((A, B))
        uses.append(eqs)
    return k, m, u, defs, uses


def walk(t, subst):
    while t[0] == "V" and t[1] in subst:
        t = subst[t[1]]
    return t


def occurs(vid, t, subst):
    t = walk(t, subst)
    if t[0] == "V":
        return t[1] == vid
    if t[0] in ("int", "bool"):
        return False
    if t[0] == "L":
        return occurs(vid, t[1], subst)
    return occurs(vid, t[1], subst) or occurs(vid, t[2], subst)


def unify(a, b, subst):
    a = walk(a, subst)
    b = walk(b, subst)
    if a[0] == "V" and b[0] == "V" and a[1] == b[1]:
        return True
    if a[0] == "V":
        if occurs(a[1], b, subst):
            return False
        subst[a[1]] = b
        return True
    if b[0] == "V":
        return unify(b, a, subst)
    if a[0] != b[0]:
        return False
    if a[0] in ("int", "bool"):
        return True
    if a[0] == "L":
        return unify(a[1], b[1], subst)
    return unify(a[1], b[1], subst) and unify(a[2], b[2], subst)


def render(t):
    if t[0] in ("int", "bool"):
        return t[0]
    if t[0] == "L":
        return "L " + render(t[1])
    return "%s %s %s" % (t[0], render(t[1]), render(t[2]))


def main():
    k, m, u, defs, uses = read_instance()
    subst = {}
    for A, B in defs:
        unify(A, B, subst)

    committed = {}  # free canonical var-id -> ground tree from the FIRST use that pins it
    for eqs in uses:
        for A, B in eqs:
            rA = walk(A, subst)
            if rA[0] == "V" and rA[1] not in committed:
                committed[rA[1]] = B  # B is always closed ground in this task

    out = [str(k)]
    for i in range(k):
        r = walk(("V", i), subst)
        if r[0] != "V":
            out.append("FIX " + render(r))
        else:
            g = committed.get(r[1], ("int",))
            out.append("FIX " + render(g))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
