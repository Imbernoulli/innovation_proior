# TIER: trivial
"""
Naive per-domain-point decoder, identity state encoding (code(s) = s), NO sharing:
every AND-indicator for a (state,symbol) point is built completely from scratch
(a fresh NOT gate for every negated literal, even if that literal was already
negated for a previous point). Output bit j = OR of the indicators of every
point whose true next-state has bit j set. This is exactly the checker's
internal baseline construction -> scores ~0.1.
"""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    S = int(next(it)); K = int(next(it))
    delta = [[0] * K for _ in range(S)]
    for s in range(S):
        for k in range(K):
            delta[s][k] = int(next(it))
    # lanes are irrelevant to circuit construction (any correct circuit works on all lanes)

    b = (S - 1).bit_length()
    m = (K - 1).bit_length()
    n = b + m

    gates = []          # list of "OP a b" / "NOT a" strings
    wire_count = [n]    # 0..b-1 state-bit wires, b..b+m-1 input-bit wires (identity encoding)

    def emit(line):
        gates.append(line)
        w = wire_count[0]
        wire_count[0] += 1
        return w

    domain = [(s, k) for s in range(S) for k in range(K)]
    indicators = []
    for (s, k) in domain:
        bits = [(s >> i) & 1 for i in range(b)] + [(k >> i) & 1 for i in range(m)]
        lits = []
        for i, v in enumerate(bits):
            if v == 1:
                lits.append(i)
            else:
                lits.append(emit("NOT %d" % i))
        cur = lits[0]
        for t in lits[1:]:
            cur = emit("AND %d %d" % (cur, t))
        indicators.append(cur)

    outwires = []
    for j in range(b):
        sel = [indicators[d] for d, (s, k) in enumerate(domain) if (delta[s][k] >> j) & 1]
        if not sel:
            w0 = indicators[0]
            nw = emit("NOT %d" % w0)
            z = emit("AND %d %d" % (w0, nw))
            outwires.append(z)
        elif len(sel) == len(domain):
            w0 = indicators[0]
            nw = emit("NOT %d" % w0)
            o = emit("OR %d %d" % (w0, nw))
            outwires.append(o)
        else:
            cur = sel[0]
            for t in sel[1:]:
                cur = emit("OR %d %d" % (cur, t))
            outwires.append(cur)

    out = []
    out.append("%d %d" % (b, m))
    out.append(str(S))
    for s in range(S):
        out.append("%d %d" % (s, s))          # identity encoding
    out.append(str(len(gates)))
    out.extend(gates)
    out.append(" ".join(str(w) for w in outwires))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
