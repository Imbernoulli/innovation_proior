#!/usr/bin/env python3
"""counter.py <in> <out> <ans>  -- deterministic scorer (format D, eval_form=flops).

The participant submits a GF(2) straight-line program (arithmetic circuit) that is
claimed to compute multiplication in GF(2^k) modulo a fixed irreducible polynomial
f(x). We FIRST verify EXACT functional equivalence to the target bilinear map (by
comparing full Boolean truth tables over all 2^(2k) input assignments -- an exact
proof, computed with big-integer bit-slicing), and only THEN count the number of
GF(2) multiplications (AND gates). Wrong function -> Ratio: 0.0.

Cost metric = number of AND gates = number of GF(2) scalar multiplications used
(the multiplicative / bilinear complexity, the dominant cost of a bit-parallel
GF(2^k) multiplier). Baseline B = schoolbook = k*k multiplications.
    ratio = min(1.0, 0.1 * B / F)          (fewer multiplications is better)
trivial (schoolbook) -> 0.1 ; a 10x reduction would cap at 1.0 (unreachable here,
since the multiplicative complexity of GF(2^k) mult is far above k*k/10).
"""
import sys

MAX_GATES = 4000        # a correct k<=9 multiplier needs < ~200 gates; cap DoS
MAX_TOKENS = 100000     # bounded read


def fail(reason):
    print("reason: " + reason)
    print("Ratio: 0.0")
    sys.exit(0)


def read_instance(path):
    with open(path) as f:
        toks = f.read().split()
    k = int(toks[0])
    m = [int(x) for x in toks[1:2 + k]]      # m_0 .. m_k
    return k, m


def var_tt(p, N):
    """Truth table (as an N-bit integer) of input variable at bit position p:
    bit x is set iff (x >> p) & 1. Built by pattern-doubling -> O(log) big-int ops."""
    block = 1 << p
    period = block << 1
    unit = ((1 << block) - 1) << block       # one period: high 'block' bits set
    val = unit
    cur = period
    while cur < N:
        val |= val << cur
        cur <<= 1
    return val & ((1 << N) - 1)


def reference_outputs(k, m, TA, TB, N):
    """Target: c = a * b mod f, per-output-bit truth tables, via bit-sliced
    schoolbook multiply + reduction (bitwise big-int ops on the truth tables)."""
    d = [0] * (2 * k - 1)
    for i in range(k):
        ai = TA[i]
        for j in range(k):
            d[i + j] ^= ai & TB[j]
    for s in range(2 * k - 2, k - 1, -1):
        ds = d[s]
        if ds:
            for i in range(k):
                if m[i]:
                    d[s - k + i] ^= ds
        d[s] = 0
    return d[:k]


def main():
    if len(sys.argv) < 3:
        fail("usage")
    inf, outf = sys.argv[1], sys.argv[2]
    k, m = read_instance(inf)
    if len(m) != k + 1 or m[k] != 1:
        fail("bad instance")

    # ---- parse participant circuit strictly ----
    try:
        with open(outf) as f:
            toks = f.read().split()
    except Exception:
        fail("cannot read output")
    if not toks:
        fail("empty output")
    if len(toks) > MAX_TOKENS:
        fail("too many tokens")

    try:
        G = int(toks[0])
    except ValueError:
        fail("gate count not an integer")
    if G < 0 or G > MAX_GATES:
        fail("gate count out of range [0,%d]" % MAX_GATES)

    n_in = 2 * k
    expected = 1 + 3 * G + k
    if len(toks) != expected:
        fail("token count %d != expected %d" % (len(toks), expected))

    gates = []            # (is_and, i, j)
    pos = 1
    for t in range(G):
        op = toks[pos]
        cur_wires = n_in + t          # wires 0 .. n_in+t-1 already defined
        try:
            i = int(toks[pos + 1]); j = int(toks[pos + 2])
        except ValueError:
            fail("gate operand not an integer")
        pos += 3
        if op == "AND":
            is_and = True
        elif op == "XOR":
            is_and = False
        else:
            fail("unknown op '%s' (want AND/XOR)" % op)
        if not (0 <= i < cur_wires) or not (0 <= j < cur_wires):
            fail("gate %d references undefined/forward wire" % t)
        gates.append((is_and, i, j))

    outs = []
    total_wires = n_in + G
    for c in range(k):
        try:
            w = int(toks[pos + c])
        except ValueError:
            fail("output wire not an integer")
        if not (0 <= w < total_wires):
            fail("output wire out of range")
        outs.append(w)

    # ---- exact equivalence via full truth tables ----
    N = 1 << (2 * k)
    TA = [var_tt(i, N) for i in range(k)]
    TB = [var_tt(k + j, N) for j in range(k)]
    cref = reference_outputs(k, m, TA, TB, N)

    TT = [0] * total_wires
    for i in range(k):
        TT[i] = TA[i]
        TT[k + i] = TB[i]
    for t, (is_and, i, j) in enumerate(gates):
        x = TT[i]; y = TT[j]
        TT[n_in + t] = (x & y) if is_and else (x ^ y)

    for c in range(k):
        if TT[outs[c]] != cref[c]:
            fail("circuit does not compute GF(2^%d) multiplication (output bit %d)" % (k, c))

    # ---- score = multiplicative complexity ----
    F = sum(1 for g in gates if g[0])          # number of AND gates
    B = k * k                                  # schoolbook baseline
    ratio = min(1.0, 0.1 * B / max(1e-9, F))
    print("ands=%d xors=%d baseline=%d" % (F, G - F, B))
    print("Ratio: %.6f" % ratio)


if __name__ == "__main__":
    main()
