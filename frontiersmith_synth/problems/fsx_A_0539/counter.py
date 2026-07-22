import sys

# Format D checker -- "build the lock that only real keys visit".
#
# Instance (<in>):   n m
#                    m lines:  <n-bit string>  <required output bit>
#   The m care vectors are the "real keys"; behaviour on every other input is FREE.
#
# Participant (<out>): a straight-line GF(2) circuit
#                    G
#                    G gate lines, each one of:
#                        AND a b | OR a b | XOR a b | NOT a | ONE | ZERO
#                    OUT w
#   Wires: inputs are wires 0..n-1 (wire i = x_i); the g-th gate (0-indexed) is
#   wire n+g and may reference only strictly-earlier wires.  Objective = G (min).
#
# Scoring:
#   1) EXACT feasibility: circuit must reproduce the required output on EVERY care
#      vector (checked with bit-parallel evaluation).  Any violation -> Ratio 0.0.
#   2) Baseline B: the checker itself detects the affine hull of the care set,
#      changes to free coordinates, computes the Reed-Muller (ANF) form there and
#      counts a naive one-AND-per-monomial circuit.  This is the affine-oblivious
#      "found the flat but synthesized naively" reference.
#   3) ratio = min(1.0, 0.1 * B / G)   (fewer gates -> higher; trivial ~ 0.1).

MAXGATES = 500000


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def read_instance(path):
    toks = open(path).read().split()
    it = iter(toks)
    try:
        n = int(next(it)); m = int(next(it))
    except Exception:
        fail("bad instance header")
    if not (1 <= n <= 40 and 1 <= m <= (1 << 20)):
        fail("instance out of range")
    xs = []; ys = []
    try:
        for _ in range(m):
            bits = next(it)
            y = int(next(it))
            if len(bits) != n or y not in (0, 1):
                fail("bad care row")
            X = 0
            for i, ch in enumerate(bits):
                if ch == '1':
                    X |= (1 << i)
                elif ch != '0':
                    fail("bad bit char")
            xs.append(X); ys.append(y)
    except StopIteration:
        fail("truncated instance")
    return n, m, xs, ys


def baseline_gate_count(n, m, xs, ys):
    """Detect affine hull, Reed-Muller (ANF) in free coords, count a naive
    one-AND-per-monomial circuit.  Deterministic -> equals trivial.py's output."""
    X0 = xs[0]
    deltas = [x ^ X0 for x in xs]
    basis = []
    for d in deltas:
        v = d
        for b in basis:
            lb = b.bit_length() - 1
            if (v >> lb) & 1:
                v ^= b
        if v:
            basis.append(v)
    B = basis[:]
    pivbits = []; r = 0
    for col in range(n - 1, -1, -1):
        sel = -1
        for row in range(r, len(B)):
            if (B[row] >> col) & 1:
                sel = row; break
        if sel < 0:
            continue
        B[r], B[sel] = B[sel], B[r]
        for row in range(len(B)):
            if row != r and (B[row] >> col) & 1:
                B[row] ^= B[r]
        pivbits.append(col); r += 1
        if r == len(B):
            break
    k = len(B)
    if k > 22:
        # safety valve (should not happen for shipped instances)
        return max(1, m)
    ytab = [0] * (1 << k)
    for p in range(m):
        xv = xs[p]
        t = 0
        for i, pb in enumerate(pivbits):
            if (xv >> pb) & 1:
                t |= (1 << i)
        ytab[t] = ys[p]
    anf = ytab[:]
    for i in range(k):
        bit = 1 << i
        for mask in range(1 << k):
            if mask & bit:
                anf[mask] ^= anf[mask ^ bit]
    nonzero = [mask for mask in range(1, 1 << k) if anf[mask]]
    ands = sum(bin(mask).count("1") - 1 for mask in nonzero)
    terms = len(nonzero) + (1 if anf[0] else 0)
    ones = 1 if anf[0] else 0
    if terms == 0:
        return 1
    xors = max(0, terms - 1)
    return ands + xors + ones


def main():
    n, m, xs, ys = read_instance(sys.argv[1])

    # bit-parallel input wire vectors (bit p = value on care point p)
    mask_all = (1 << m) - 1
    wires = [0] * n
    for i in range(n):
        w = 0
        for p in range(m):
            if (xs[p] >> i) & 1:
                w |= (1 << p)
        wires[i] = w
    yvec = 0
    for p in range(m):
        if ys[p]:
            yvec |= (1 << p)

    out = open(sys.argv[2]).read().split()
    if not out:
        fail("empty output")
    idx = 0
    try:
        G = int(out[idx]); idx += 1
    except Exception:
        fail("bad G")
    if G < 0 or G > MAXGATES:
        fail("G out of range")

    def rdint():
        nonlocal idx
        v = int(out[idx]); idx += 1  # int() rejects nan/inf -> ValueError -> caught
        return v

    try:
        for g in range(G):
            cur = n + g
            op = out[idx]; idx += 1
            if op in ("AND", "OR", "XOR"):
                a = rdint(); b = rdint()
                if not (0 <= a < cur and 0 <= b < cur):
                    fail("wire ref out of range")
                va, vb = wires[a], wires[b]
                if op == "AND":
                    wires.append(va & vb)
                elif op == "OR":
                    wires.append(va | vb)
                else:
                    wires.append(va ^ vb)
            elif op == "NOT":
                a = rdint()
                if not (0 <= a < cur):
                    fail("wire ref out of range")
                wires.append((~wires[a]) & mask_all)
            elif op == "ONE":
                wires.append(mask_all)
            elif op == "ZERO":
                wires.append(0)
            else:
                fail("unknown op '%s'" % op)
        if out[idx] != "OUT":
            fail("missing OUT")
        idx += 1
        outw = rdint()
    except (IndexError, ValueError):
        fail("malformed / non-finite token")
    if not (0 <= outw < n + G):
        fail("OUT wire out of range")
    if idx != len(out):
        fail("trailing tokens")

    if (wires[outw] & mask_all) != yvec:
        fail("circuit disagrees with a real key")

    B = baseline_gate_count(n, m, xs, ys)
    F = G if G > 0 else 1
    ratio = min(1.0, 0.1 * B / max(1e-9, F))
    print("G=%d B=%d Ratio: %.6f" % (G, B, ratio))


if __name__ == "__main__":
    main()
