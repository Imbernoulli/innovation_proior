# TIER: strong
#
# Strong reference: the automaton's STATE SHARING becomes circuit
# SUBEXPRESSION SHARING. Instead of expanding the accepted strings into a
# sum-of-products (which blows up with |L|), we:
#   1) run subset construction level by level (positions 0..n), keeping only
#      subsets that are both reachable from the start AND can still reach an
#      accept state within the remaining positions (bi-directional trim);
#   2) build the predicate backward: value(i, Q) = "from subset Q at position
#      i, the remaining bits can drive the NFA to an accept", with
#      value(i, Q) = (~x_i AND value(i+1, d(Q,0))) OR (x_i AND value(i+1, d(Q,1)));
#   3) hash-cons EVERY gate (global common-subexpression elimination with
#      constant folding and commutative canonicalization), so two subsets with
#      identical future behaviour literally reuse the same wires -- the DFA's
#      merged states turn into merged subcircuits.
# Gate cost ~ O(n * #live subsets), independent of |L|.
#
# The builder below is imported by ../gen.py to predict this solution's exact
# gate count when calibrating instances; keep it deterministic (sorted
# iteration everywhere) and keep this file free of prints at import time.

import sys


def build_circuit(n, S, q0, accepts, trans):
    """trans: dict[(p,b)] -> tuple of target states (sorted).
    Returns (gates, out_wire) where gates is a list of (op, a, b);
    b is None for CONST/NOT. Wire i < n is input bit i; gate g makes wire n+g."""
    cons = {}
    gates = []

    def emit(key):
        if key in cons:
            return cons[key]
        w = n + len(gates)
        gates.append(key)
        cons[key] = w
        return w

    C0 = emit(("CONST", 0, None))
    C1 = emit(("CONST", 1, None))

    def NOT(a):
        if a == C0:
            return C1
        if a == C1:
            return C0
        return emit(("NOT", a, None))

    def AND(a, b):
        if a == C0 or b == C0:
            return C0
        if a == C1:
            return b
        if b == C1:
            return a
        if a == b:
            return a
        if a > b:
            a, b = b, a
        return emit(("AND", a, b))

    def OR(a, b):
        if a == C1 or b == C1:
            return C1
        if a == C0:
            return b
        if b == C0:
            return a
        if a == b:
            return a
        if a > b:
            a, b = b, a
        return emit(("OR", a, b))

    notx = [NOT(i) for i in range(n)]

    tr = {k: frozenset(v) for k, v in trans.items()}
    empty = frozenset()
    acc = frozenset(accepts)

    def step(sub, b):
        s = set()
        for p in sub:
            s |= tr.get((p, b), empty)
        return frozenset(s)

    # forward reachable subsets per level
    reach = [set() for _ in range(n + 1)]
    reach[0].add(frozenset([q0]))
    for i in range(n):
        for sub in reach[i]:
            reach[i + 1].add(step(sub, 0))
            reach[i + 1].add(step(sub, 1))

    # backward "can still accept within remaining positions" trim
    can = [set() for _ in range(n + 1)]
    can[n] = set(s for s in reach[n] if s & acc)
    for i in range(n - 1, -1, -1):
        for sub in reach[i]:
            if step(sub, 0) in can[i + 1] or step(sub, 1) in can[i + 1]:
                can[i].add(sub)

    def key(s):
        return (len(s), sorted(s))

    val = {}
    for sub in sorted(can[n], key=key):
        val[(n, sub)] = C1
    for i in range(n - 1, -1, -1):
        for sub in sorted(can[i], key=key):
            g = C0
            t0 = val.get((i + 1, step(sub, 0)))
            if t0 is not None:
                g = OR(g, AND(notx[i], t0))
            t1 = val.get((i + 1, step(sub, 1)))
            if t1 is not None:
                g = OR(g, AND(i, t1))
            val[(i, sub)] = g

    start = frozenset([q0])
    out = val.get((0, start), C0)
    return gates, out


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
    gates, out = build_circuit(n, S, q0, accepts, trans)
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
