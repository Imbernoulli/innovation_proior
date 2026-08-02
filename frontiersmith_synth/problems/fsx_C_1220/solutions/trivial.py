# TIER: trivial
"""Blind baseline: solves the DEFINITION constraints (so it stays sound), but
never looks at the held-out uses at all -- every variable the definitions leave
free is just fixed to `int` regardless of what any call site wants.
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
    # trivial doesn't need the use blocks at all -- but still must consume/skip
    # them correctly (nothing else to do with a broken/short read here).
    for _ in range(u):
        tr.expect("USE")
        tr.next_int()
        q = tr.next_int()
        for _ in range(q):
            tr.parse_type()
            tr.expect("=")
            tr.parse_type()
    return k, m, defs


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
    k, m, defs = read_instance()
    subst = {}
    for A, B in defs:
        unify(A, B, subst)

    out = [str(k)]
    for i in range(k):
        r = walk(("V", i), subst)
        if r[0] != "V":
            out.append("FIX " + render(r))
        else:
            out.append("FIX int")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
