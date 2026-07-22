# TIER: greedy
"""
Textbook Boolean-circuit synthesis: compile each output bit to its Algebraic
Normal Form (Reed-Muller / Moebius transform over GF(2)) and realize it as a
XOR-of-AND-monomials circuit. This is the standard "derive the minimal XOR/AND
canonical form" recipe a competent engineer reaches for -- but it is applied to
the states AS GIVEN (identity numbering, code(s) = s). No search over how the
states could be relabeled is attempted, so on instances whose dynamics are only
linear under a DIFFERENT (hidden) labeling, the ANF stays dense and the circuit
stays big.
"""
import sys


def anf_transform(truth, n):
    a = truth[:]
    size = 1 << n
    for i in range(n):
        bit = 1 << i
        for x in range(size):
            if x & bit:
                a[x] ^= a[x ^ bit]
    return a


def emit_bit_from_anf(anf, n, emit):
    monomials = [mask for mask in range(1, 1 << n) if anf[mask]]
    term_wires = []
    for mask in monomials:
        bits = [i for i in range(n) if (mask >> i) & 1]
        if len(bits) == 1:
            term_wires.append(bits[0])
        else:
            cur = bits[0]
            for t in bits[1:]:
                cur = emit("AND %d %d" % (cur, t))
            term_wires.append(cur)
    if not term_wires:
        w0 = 0
        nw = emit("NOT %d" % w0)
        if anf[0] == 0:
            return emit("AND %d %d" % (w0, nw))
        else:
            return emit("OR %d %d" % (w0, nw))
    cur = term_wires[0]
    for t in term_wires[1:]:
        cur = emit("XOR %d %d" % (cur, t))
    if anf[0] == 1:
        cur = emit("NOT %d" % cur)
    return cur


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    S = int(next(it)); K = int(next(it))
    delta = [[0] * K for _ in range(S)]
    for s in range(S):
        for k in range(K):
            delta[s][k] = int(next(it))

    b = (S - 1).bit_length()
    m = (K - 1).bit_length()
    n = b + m

    code_of = list(range(S))       # identity encoding: no representation search

    Y = [0] * (1 << n)
    for s in range(S):
        for k in range(K):
            x = code_of[s] | (k << b)
            Y[x] = code_of[delta[s][k]]

    gates = []
    wire_count = [n]

    def emit(line):
        gates.append(line)
        w = wire_count[0]
        wire_count[0] += 1
        return w

    outwires = []
    for j in range(b):
        truth = [(Y[x] >> j) & 1 for x in range(1 << n)]
        anf = anf_transform(truth, n)
        outwires.append(emit_bit_from_anf(anf, n, emit))

    out = []
    out.append("%d %d" % (b, m))
    out.append(str(S))
    for s in range(S):
        out.append("%d %d" % (s, code_of[s]))
    out.append(str(len(gates)))
    out.extend(gates)
    out.append(" ".join(str(w) for w in outwires))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
