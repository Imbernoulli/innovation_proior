#!/usr/bin/env python3
# Deterministic checker for fsx_S_0561 (format D, minimize gate count).
#
#   python3 counter.py <in> <out> <ans>
#
# <in>  : the obfuscated target circuit (defines the target boolean function f)
# <out> : the participant's circuit (must compute the SAME f)
# <ans> : ignored placeholder
#
# Procedure:
#   1. parse the target circuit, compute its truth table T over all 2^n inputs.
#   2. parse the participant circuit STRICTLY; any schema/range/finiteness
#      violation  -> Ratio: 0.0
#   3. compute the participant truth table S; if S != T -> Ratio: 0.0
#   4. F = number of participant gates; B = number of target gates (the trivial
#      "resubmit the target as-is" baseline). Minimization:
#         sc = min(1000, 100 * B / max(1,F));  Ratio = sc/1000
#      trivial (echo the target) -> 0.1 ; a 10x smaller circuit caps at 1.0.

import sys

MAX_GATES = 200000
OPS2 = ('AND', 'OR', 'XOR')


def _fail(msg):
    print("reason: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def parse_strict(text, expect_n=None):
    """Parse a circuit; return (n, gates, out) or raise ValueError."""
    toks = text.split()
    if not toks:
        raise ValueError("empty")
    it = iter(toks)

    def nxt():
        return next(it)

    n = int(nxt())
    g = int(nxt())
    if n < 1 or n > 24:
        raise ValueError("n out of range")
    if g < 0 or g > MAX_GATES:
        raise ValueError("g out of range")
    if expect_n is not None and n != expect_n:
        raise ValueError("n mismatch")
    gates = []
    for k in range(g):
        op = nxt()
        idd = n + k
        if op in OPS2:
            a = int(nxt())
            b = int(nxt())
            if not (0 <= a < idd) or not (0 <= b < idd):
                raise ValueError("operand out of range at gate %d" % k)
            gates.append((op, a, b))
        elif op == 'NOT':
            a = int(nxt())
            if not (0 <= a < idd):
                raise ValueError("operand out of range at gate %d" % k)
            gates.append((op, a, -1))
        elif op in ('CONST0', 'CONST1'):
            gates.append((op, -1, -1))
        else:
            raise ValueError("bad op %r" % op)
    key = nxt()
    if key != 'OUTPUT':
        raise ValueError("missing OUTPUT")
    out = int(nxt())
    if not (0 <= out < n + g):
        raise ValueError("output node out of range")
    # no trailing garbage
    try:
        nxt()
        raise ValueError("trailing tokens")
    except StopIteration:
        pass
    return n, gates, out


def input_tts(n):
    N = 1 << n
    M = (1 << N) - 1
    tts = []
    for i in range(n):
        half = 1 << i
        col = ((1 << half) - 1) << half
        w = 2 * half
        while w < N:
            col |= col << w
            w <<= 1
        tts.append(col & M)
    return tts, M


def evaluate(n, gates, out):
    tts, M = input_tts(n)
    tt = list(tts)
    for (op, a, b) in gates:
        if op == 'CONST0':
            tt.append(0)
        elif op == 'CONST1':
            tt.append(M)
        elif op == 'NOT':
            tt.append(tt[a] ^ M)
        elif op == 'AND':
            tt.append(tt[a] & tt[b])
        elif op == 'OR':
            tt.append(tt[a] | tt[b])
        else:  # XOR
            tt.append(tt[a] ^ tt[b])
    return tt[out] & M


def main():
    inp, outp = sys.argv[1], sys.argv[2]
    with open(inp) as f:
        n_t, gates_t, out_t = parse_strict(f.read())
    T = evaluate(n_t, gates_t, out_t)
    B = len(gates_t)

    with open(outp) as f:
        sub_text = f.read()
    try:
        n_s, gates_s, out_s = parse_strict(sub_text, expect_n=n_t)
    except (ValueError, StopIteration) as e:
        _fail("invalid submission (%s)" % e)

    S = evaluate(n_s, gates_s, out_s)
    if S != T:
        _fail("submission not equivalent to target function")

    F = len(gates_s)
    sc = min(1000.0, 100.0 * B / max(1, F))
    print("gates_target=%d gates_submission=%d equivalent=yes" % (B, F))
    print("Ratio: %.6f" % (sc / 1000.0))


if __name__ == '__main__':
    main()
