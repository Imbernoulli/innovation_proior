import os
import sys

# Format D checker -- smallest Boolean circuit for a regular predicate.
#
# Instance (<in>):  a small NFA over {0,1} describing a predicate on n-bit
#   strings (the automaton reads bit 0 first):
#       n S q0 K M
#       a_1 ... a_K          (accept states, distinct)
#       M lines: p b q       (transition from p to q on bit b in {0,1})
#   The accepted language L = { v in [0,2^n) : the NFA has an accepting run
#   on the n bits of v, bit 0 first }.
#
# Participant (<out>): a straight-line Boolean circuit
#       G
#       G gate lines:  CONST c (c in {0,1}) | NOT a | AND a b | OR a b | XOR a b
#       OUT w
#   Wires: inputs are wires 0..n-1 (wire i = bit i of the tested string); the
#   g-th gate line (0-indexed) creates wire n+g and may reference only
#   STRICTLY earlier wires. OUT names the single output wire.
#
# Scoring (deterministic, exact):
#   1) EXACT equivalence gate: the circuit's truth table over all 2^n inputs
#      must equal the language's characteristic table. Tables are compared as
#      exact 2^n-bit integers (bit-parallel evaluation with big-int bitwise
#      ops -- no floats anywhere, bit-for-bit deterministic). Any mismatch,
#      malformed token, non-integer/non-finite token, out-of-range wire,
#      out-of-range literal, forward reference, or trailing tokens -> 0.0.
#   2) Objective (minimize) = G, the gate count.
#      Baseline B = the exact gate count of the naive sum-of-products
#      construction the checker builds itself from the language:
#          B = sum_{v in L} z(v) + A*(n-1) + (A-1)
#      where A = |L| and z(v) = number of 0-bits of v (one fresh NOT per
#      0-bit per accepted string, an AND chain of n-1 gates per string, and
#      A-1 OR gates to combine them). No sharing of any kind.
#          ratio = min(1.0, 0.1 * B / max(1, G))

MAXGATES = 400000
MAXN = 13
MAXS = 64
MAXM = 8192
MAXOUTBYTES = 64 * 1024 * 1024


def fail(reason):
    print("infeasible: %s" % reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    try:
        toks = open(path).read().split()
    except Exception:
        fail("cannot read instance")
    pos = [0]

    def nxt():
        if pos[0] >= len(toks):
            fail("truncated instance")
        t = toks[pos[0]]
        pos[0] += 1
        try:
            return int(t)
        except Exception:
            fail("non-integer token in instance")

    n = nxt(); S = nxt(); q0 = nxt(); K = nxt(); M = nxt()
    if not (1 <= n <= MAXN):
        fail("instance n out of range")
    if not (1 <= S <= MAXS):
        fail("instance S out of range")
    if not (0 <= q0 < S):
        fail("instance q0 out of range")
    if not (1 <= K <= S):
        fail("instance K out of range")
    if not (0 <= M <= MAXM):
        fail("instance M out of range")
    accepts = set()
    for _ in range(K):
        a = nxt()
        if not (0 <= a < S):
            fail("accept state out of range")
        accepts.add(a)
    if len(accepts) != K:
        fail("duplicate accept states")
    trans = {}
    for _ in range(M):
        p = nxt(); b = nxt(); q = nxt()
        if not (0 <= p < S and 0 <= q < S and b in (0, 1)):
            fail("bad transition line")
        trans.setdefault((p, b), set()).add(q)
    return n, S, q0, accepts, trans


def language_table(n, q0, accepts, trans):
    """Characteristic truth table of the accepted language as a 2^n-bit
    integer (bit v set iff the NFA accepts v's bits, bit 0 read first).
    Also returns A = |L| and sumz = total number of 0-bits over L."""
    N = 1 << n
    table = 0
    A = 0
    sumz = 0
    acc = frozenset(accepts)
    empty = frozenset()
    tr = {k: frozenset(v) for k, v in trans.items()}
    for v in range(N):
        cur = frozenset([q0])
        for i in range(n):
            b = (v >> i) & 1
            nxt = set()
            for p in cur:
                nxt |= tr.get((p, b), empty)
            cur = frozenset(nxt)
        if cur & acc:
            table |= (1 << v)
            A += 1
            sumz += n - bin(v).count("1")
    return table, A, sumz


def parse_program(path, n):
    try:
        if os.path.getsize(path) > MAXOUTBYTES:
            fail("output too large")
        out = open(path).read().split()
    except Exception:
        fail("cannot read output")
    if not out:
        fail("empty output")
    idx = [0]

    def nxt():
        if idx[0] >= len(out):
            fail("truncated output")
        t = out[idx[0]]
        idx[0] += 1
        return t

    def toint():
        t = nxt()
        try:
            return int(t)
        except Exception:
            fail("non-integer / non-finite token '%s'" % t)

    G = toint()
    if not (0 <= G <= MAXGATES):
        fail("G out of range")
    gates = []
    for g in range(G):
        cur = n + g
        op = nxt()
        if op == "CONST":
            c = toint()
            if c not in (0, 1):
                fail("CONST literal out of range")
            gates.append((op, c, None))
        elif op == "NOT":
            a = toint()
            if not (0 <= a < cur):
                fail("NOT wire reference out of range / not strictly earlier")
            gates.append((op, a, None))
        elif op in ("AND", "OR", "XOR"):
            a = toint(); b = toint()
            if not (0 <= a < cur and 0 <= b < cur):
                fail("wire reference out of range / not strictly earlier")
            gates.append((op, a, b))
        else:
            fail("unknown opcode '%s'" % op)
    tok = nxt()
    if tok != "OUT":
        fail("missing OUT")
    w = toint()
    if not (0 <= w < n + G):
        fail("OUT wire out of range")
    if idx[0] != len(out):
        fail("trailing tokens")
    return G, gates, w


def evaluate(n, gates, outw):
    """Bit-parallel exact evaluation: every wire carries its full 2^n-bit
    truth table as a big integer. Pure integer ops -> deterministic."""
    N = 1 << n
    mask = (1 << N) - 1
    wires = []
    for i in range(n):
        blk = 1 << i
        ones = ((1 << blk) - 1) << blk
        t = 0
        for start in range(0, N, blk * 2):
            t |= ones << start
        wires.append(t)
    for op, a, b in gates:
        if op == "CONST":
            wires.append(mask if a else 0)
        elif op == "NOT":
            wires.append(mask ^ wires[a])
        elif op == "AND":
            wires.append(wires[a] & wires[b])
        elif op == "OR":
            wires.append(wires[a] | wires[b])
        else:  # XOR
            wires.append(wires[a] ^ wires[b])
    return wires[outw]


def main():
    n, S, q0, accepts, trans = read_instance(sys.argv[1])
    table, A, sumz = language_table(n, q0, accepts, trans)
    G, gates, outw = parse_program(sys.argv[2], n)
    got = evaluate(n, gates, outw)
    if got != table:
        fail("circuit does not compute the accepted language exactly")
    B = sumz + A * (n - 1) + max(0, A - 1)
    B = max(1, B)
    F = max(1, G)
    ratio = min(1.0, 0.1 * B / F)
    print("G=%d B=%d A=%d" % (G, B, A))
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
