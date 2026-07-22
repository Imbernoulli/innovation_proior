# TIER: trivial
#
# Trivial baseline: enumerate every accepted n-bit string by simulating the
# NFA, and emit one fully independent product term per accepted string -- a
# FRESH NOT gate for every 0-bit of every string (no sharing at all), an
# AND chain of n-1 gates per string, and a final OR chain of A-1 gates.
# This is exactly the checker's internal baseline construction, so its gate
# count equals B by definition.

import sys


def main():
    toks = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        t = int(toks[pos]); pos += 1
        return t

    n = nxt(); S = nxt(); q0 = nxt(); K = nxt(); M = nxt()
    accepts = frozenset(nxt() for _ in range(K))
    trans = {}
    for _ in range(M):
        p = nxt(); b = nxt(); q = nxt()
        trans.setdefault((p, b), set()).add(q)
    tr = {k: frozenset(v) for k, v in trans.items()}
    empty = frozenset()

    acc_strings = []
    for v in range(1 << n):
        cur = frozenset([q0])
        for i in range(n):
            b = (v >> i) & 1
            s = set()
            for p in cur:
                s |= tr.get((p, b), empty)
            cur = frozenset(s)
        if cur & accepts:
            acc_strings.append(v)

    gates = []

    def emit(op, a, b=None):
        gates.append((op, a, b))
        return n + len(gates) - 1

    if not acc_strings:
        w = emit("CONST", 0)
    else:
        termw = []
        for v in acc_strings:
            lits = []
            for i in range(n):
                if (v >> i) & 1:
                    lits.append(i)
                else:
                    lits.append(emit("NOT", i))   # fresh NOT, never shared
            w = lits[0]
            for lit in lits[1:]:
                w = emit("AND", w, lit)
            termw.append(w)
        w = termw[0]
        for t in termw[1:]:
            w = emit("OR", w, t)

    lines = [str(len(gates))]
    for op, a, b in gates:
        if op == "CONST":
            lines.append("CONST %d" % a)
        elif op == "NOT":
            lines.append("NOT %d" % a)
        else:
            lines.append("%s %d %d" % (op, a, b))
    lines.append("OUT %d" % w)
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
