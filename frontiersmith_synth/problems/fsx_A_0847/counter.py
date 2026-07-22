#!/usr/bin/env python3
"""
counter.py <in> <out> <ans> -- checker for bitsliced-transition-synthesis.

Verifies the submitted (state-encoding, bitwise straight-line program) EXACTLY
reproduces the FSM's next-state on all 64 lockstep lanes given in <in>, THEN
scores by op-count against an internal naive baseline B (a from-scratch,
no-sharing sum-of-products decoder in the SAME public 0..S-1 numbering).

Ratio = min(1, 0.1 * B / ops_submitted)   (fewer gates -> higher ratio)
"""
import sys

MASK64 = (1 << 64) - 1
MAX_GATES = 20000
OPS2 = {"AND", "OR", "XOR"}
OPS1 = {"NOT"}


def fail(msg):
    print("INFEASIBLE: %s" % msg)
    print("Ratio: 0.0")
    sys.exit(0)


def read_ints_line(f):
    line = f.readline()
    if line == "":
        return None
    return line.split()


def parse_input(inf):
    with open(inf) as f:
        toks = f.read().split()
    it = iter(toks)
    try:
        S = int(next(it)); K = int(next(it))
        delta = [[0] * K for _ in range(S)]
        for s in range(S):
            for k in range(K):
                delta[s][k] = int(next(it))
        lanes = []
        for _ in range(64):
            s = int(next(it)); k = int(next(it))
            lanes.append((s, k))
    except (StopIteration, ValueError):
        raise RuntimeError("bad input file")
    b = (S - 1).bit_length()
    m = (K - 1).bit_length()
    if (1 << b) != S or (1 << m) != K:
        raise RuntimeError("input S,K not powers of two")
    return S, K, b, m, delta, lanes


def naive_baseline_ops(S, K, b, m, delta):
    """Same algorithm as solutions/trivial.py: per-domain-point AND-indicator built
    from scratch (fresh NOT per negated literal, no cross-indicator sharing), then a
    plain OR-select per output bit. Returns the gate COUNT only (fast, no wire lists)."""
    n = b + m
    domain = [(s, k) for s in range(S) for k in range(K)]
    gate_count = 0
    ind_negcount = []  # number of NOT gates used to build indicator d
    for (s, k) in domain:
        bits = [(s >> i) & 1 for i in range(b)] + [(k >> i) & 1 for i in range(m)]
        zeros = sum(1 for v in bits if v == 0)
        gate_count += zeros                      # fresh NOT per zero literal
        gate_count += max(0, n - 1)               # AND-chain over n literals
        ind_negcount.append(True)
    domsz = len(domain)
    for j in range(b):
        sel = 0
        for (s, k) in domain:
            if (delta[s][k] >> j) & 1:
                sel += 1
        if sel == 0 or sel == domsz:
            gate_count += 2                       # constant realize: NOT + AND/OR
        else:
            gate_count += max(0, sel - 1)         # OR-chain
    return gate_count


def main():
    if len(sys.argv) < 3:
        fail("bad invocation")
    inf, outf = sys.argv[1], sys.argv[2]

    try:
        S, K, b, m, delta, lanes = parse_input(inf)
    except Exception as e:
        # malformed input is an authoring bug, not a scoring case; fail loudly but safely
        print("Ratio: 0.0")
        sys.exit(0)

    try:
        with open(outf) as f:
            content = f.read()
    except Exception:
        fail("cannot read output")

    toks = content.split()
    pos = [0]

    def nxt():
        if pos[0] >= len(toks):
            raise ValueError("EOF")
        t = toks[pos[0]]
        pos[0] += 1
        return t

    def nxt_int(lo=None, hi=None):
        t = nxt()
        try:
            v = int(t)
        except ValueError:
            raise ValueError("not an int: %r" % t)
        if v != v or abs(v) == float("inf"):
            raise ValueError("non-finite")
        if lo is not None and v < lo:
            raise ValueError("out of range (too low)")
        if hi is not None and v > hi:
            raise ValueError("out of range (too high)")
        return v

    try:
        b_sub = nxt_int(0, 64)
        m_sub = nxt_int(0, 64)
        if b_sub != b or m_sub != m:
            fail("b,m mismatch: got %d,%d expected %d,%d" % (b_sub, m_sub, b, m))

        S_sub = nxt_int(0, 1 << 20)
        if S_sub != S:
            fail("state count mismatch")

        code_of = {}
        seen_codes = set()
        for _ in range(S):
            s_idx = nxt_int(0, S - 1)
            code = nxt_int(0, (1 << b) - 1)
            if s_idx in code_of:
                fail("duplicate state in encoding")
            if code in seen_codes:
                fail("duplicate code in encoding (not a bijection)")
            code_of[s_idx] = code
            seen_codes.add(code)
        if len(code_of) != S or set(code_of.keys()) != set(range(S)):
            fail("encoding does not cover all states")
        if seen_codes != set(range(1 << b)):
            fail("encoding codes are not a bijection onto 0..2^b-1")
        code_to_state = {c: s for s, c in code_of.items()}

        P = nxt_int(0, MAX_GATES)
        gates = []
        wire_count = b + m
        for _ in range(P):
            op = nxt()
            if op not in OPS1 and op not in OPS2:
                fail("unknown opcode %r" % op)
            if op in OPS1:
                a = nxt_int(0, wire_count - 1)
                gates.append((op, a, None))
            else:
                a = nxt_int(0, wire_count - 1)
                bb = nxt_int(0, wire_count - 1)
                gates.append((op, a, bb))
            wire_count += 1

        outwires = []
        for _ in range(b):
            w = nxt_int(0, wire_count - 1)
            outwires.append(w)

        if pos[0] != len(toks):
            fail("trailing garbage after the expected %d tokens (%d extra)" %
                 (pos[0], len(toks) - pos[0]))
    except ValueError as e:
        fail("parse error: %s" % e)

    # ---- simulate on the 64 lanes ----
    wires = [0] * wire_count
    for i in range(b):
        word = 0
        for lane, (s, k) in enumerate(lanes):
            bit = (code_of[s] >> i) & 1
            word |= (bit << lane)
        wires[i] = word
    for i in range(m):
        word = 0
        for lane, (s, k) in enumerate(lanes):
            bit = (k >> i) & 1
            word |= (bit << lane)
        wires[b + i] = word

    idx = b + m
    for (op, a, bb) in gates:
        if op == "NOT":
            wires[idx] = (~wires[a]) & MASK64
        elif op == "AND":
            wires[idx] = wires[a] & wires[bb]
        elif op == "OR":
            wires[idx] = wires[a] | wires[bb]
        elif op == "XOR":
            wires[idx] = wires[a] ^ wires[bb]
        idx += 1

    for lane, (s, k) in enumerate(lanes):
        code = 0
        for i in range(b):
            bit = (wires[outwires[i]] >> lane) & 1
            code |= (bit << i)
        pred_state = code_to_state.get(code)
        if pred_state is None:
            fail("output code %d at lane %d is not a valid state code" % (code, lane))
        true_state = delta[s][k]
        if pred_state != true_state:
            fail("mismatch at lane %d: state=%d sym=%d expected=%d got=%d" %
                 (lane, s, k, true_state, pred_state))

    F = P
    B = naive_baseline_ops(S, K, b, m, delta)
    ratio = min(1.0, 0.1 * B / max(1e-9, F))
    print("ops=%d baseline=%d" % (F, B))
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
