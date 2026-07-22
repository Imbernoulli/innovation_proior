# TIER: greedy
#
# Greedy reference: the textbook two-level recipe. Enumerate the accepted
# strings by simulating the NFA on every n-bit input, then run classic
# Quine-McCluskey prime-implicant expansion + essential-prime extraction +
# greedy set cover, and emit the resulting minimized DNF as a two-level
# circuit (shared NOT gates for the n inputs, one AND chain per term, one OR
# chain on top). This is the obvious "gate minimization" approach: it only
# ever shares the input negations and whatever the two-level DNF structure
# provides -- it can never share the multi-level subexpressions that the
# automaton's state graph exposes.
#
# minimize_cover() is imported by ../gen.py for calibration; keep it
# deterministic (sorted iteration, fixed tie-breaks) and print-free.

import sys


def accepted_strings(n, q0, accepts, trans):
    tr = {k: frozenset(v) for k, v in trans.items()}
    empty = frozenset()
    acc = frozenset(accepts)
    res = []
    for v in range(1 << n):
        cur = frozenset([q0])
        for i in range(n):
            b = (v >> i) & 1
            nxt = set()
            for p in cur:
                nxt |= tr.get((p, b), empty)
            cur = frozenset(nxt)
        if cur & acc:
            res.append(v)
    return res


def popcount(x):
    return bin(x).count("1")


def minimize_cover(n, minterms):
    """Quine-McCluskey: returns a deterministic list of implicants
    (val, mask) covering exactly `minterms` (mask = cared bits; val has zeros
    in don't-care bits)."""
    minterms = sorted(set(minterms))
    FULL = (1 << n) - 1
    P = set((m, FULL) for m in minterms)
    primes = set()
    while P:
        merged = set()
        used = set()
        for (v, m) in sorted(P):
            mm = m
            d = 0
            while mm:
                if mm & 1:
                    partner = (v ^ (1 << d), m)
                    if partner in P:
                        used.add((v, m))
                        used.add(partner)
                        merged.add((v & ~(1 << d), m & ~(1 << d)))
                mm >>= 1
                d += 1
        for t in P:
            if t not in used:
                primes.add(t)
        if len(merged) > 400000:   # deterministic blow-up guard: stop expanding
            primes |= merged
            break
        P = merged
    primes = sorted(primes)

    def covers(p, m):
        v, mk = p
        return (m & mk) == v

    cover = {}
    for p in primes:
        c = set(m for m in minterms if covers(p, m))
        if c:
            cover[p] = c
    # essential primes
    chosen = []
    uncovered = set(minterms)
    cnt = {}
    for m in minterms:
        cnt[m] = [p for p in primes if p in cover and m in cover[p]]
    for m in minterms:
        if len(cnt[m]) == 1:
            p = cnt[m][0]
            if p not in chosen:
                chosen.append(p)
    for p in chosen:
        uncovered -= cover[p]
    chosen.sort()
    # greedy set cover: max new coverage; ties -> fewer cared bits (more
    # general term); then lexicographic (val, mask) for full determinism
    while uncovered:
        best = None
        bestkey = None
        for p in primes:
            if p not in cover or p in chosen:
                continue
            new = len(cover[p] & uncovered)
            if new == 0:
                continue
            k = (-new, popcount(p[1]), p[0], p[1])
            if bestkey is None or k < bestkey:
                bestkey = k
                best = p
        if best is None:
            # no prime covers the rest (cannot happen); cover with minterms
            for m in sorted(uncovered):
                chosen.append((m, FULL))
            break
        chosen.append(best)
        uncovered -= cover[best]
    chosen.sort()
    return chosen


def emit_circuit(n, terms):
    """Two-level circuit for DNF `terms`: n shared NOT gates, AND chain per
    term, OR chain on top. Returns (gates, out_wire)."""
    gates = []

    def emit(op, a, b=None):
        gates.append((op, a, b))
        return n + len(gates) - 1

    notx = [emit("NOT", i) for i in range(n)]
    termw = []
    for (v, m) in terms:
        lits = []
        for i in range(n):
            if (m >> i) & 1:
                lits.append(i if ((v >> i) & 1) else notx[i])
        if not lits:
            # tautology term: language is everything
            w = emit("CONST", 1)
        else:
            w = lits[0]
            for lit in lits[1:]:
                w = emit("AND", w, lit)
        termw.append(w)
    w = termw[0]
    for t in termw[1:]:
        w = emit("OR", w, t)
    return gates, w


def parse_instance(data):
    toks = data.split()
    pos = 0

    def nxt():
        nonlocal pos
        t = int(toks[pos]); pos += 1
        return t

    n = nxt(); S = nxt(); q0 = nxt(); K = nxt(); M = nxt()
    accepts = [nxt() for _ in range(K)]
    trans = {}
    for _ in range(M):
        p = nxt(); b = nxt(); q = nxt()
        trans.setdefault((p, b), set()).add(q)
    trans = {k: tuple(sorted(v)) for k, v in trans.items()}
    return n, S, q0, accepts, trans


def main():
    n, S, q0, accepts, trans = parse_instance(sys.stdin.read())
    minterms = accepted_strings(n, q0, accepts, trans)
    if not minterms:
        sys.stdout.write("1\nCONST 0\nOUT %d\n" % n)
        return
    terms = minimize_cover(n, minterms)
    gates, out = emit_circuit(n, terms)
    lines = [str(len(gates))]
    for op, a, b in gates:
        if op == "CONST":
            lines.append("CONST %d" % a)
        elif op == "NOT":
            lines.append("NOT %d" % a)
        else:
            lines.append("%s %d %d" % (op, a, b))
    lines.append("OUT %d" % out)
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
