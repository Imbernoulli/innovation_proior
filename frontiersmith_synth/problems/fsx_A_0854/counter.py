import sys

# Format D checker -- minimal-gate shared-XOR-circuit for many linear (parity)
# forms over GF(2).
#   1) Parse R target bit-masks (each a set of variable indices) from <in>.
#   2) Parse participant's straight-line XOR circuit from <out>:
#         G
#         G gate lines "a b"   (wire C+i = wire[a] XOR wire[b], i = 1..G)
#         R lines "w_i"        (claims wire w_i computes target i)
#   3) EXACT-equivalence gate: simulate every wire over GF(2) (python big-int
#      bitmask); every claimed w_i must match target i exactly.
#   4) Objective (minimize) = G. Baseline B = sum(k_i - 1): the gate count of
#      building every target independently, bit by bit, with zero sharing.
#      Ratio = min(1, 0.1 * B / G).

MAXG = 20000
MAX_R = 500
MAX_C = 200


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    iin = iter(inp)
    try:
        R = int(next(iin)); C = int(next(iin))
    except Exception:
        fail("bad header")
    if not (1 <= R <= MAX_R and 1 <= C <= MAX_C):
        fail("bad dims")

    rows = []
    try:
        for _ in range(R):
            k = int(next(iin))
            if not (2 <= k <= C):
                fail("bad row weight")
            idxs = [int(next(iin)) for _ in range(k)]
            if any(not (1 <= v <= C) for v in idxs) or len(set(idxs)) != k:
                fail("bad row indices")
            mask = 0
            for v in idxs:
                mask |= (1 << (v - 1))
            rows.append(mask)
    except Exception:
        fail("truncated/bad input")

    # ---- parse participant output ----
    iout = iter(out)
    try:
        G = int(next(iout))
    except Exception:
        fail("bad G")
    if G < 0 or G > MAXG:
        fail("G out of range")

    wire_val = [0] * (C + G + 1)  # 1-indexed; index 0 unused
    for v in range(1, C + 1):
        wire_val[v] = 1 << (v - 1)

    try:
        for i in range(1, G + 1):
            a = int(next(iout)); b = int(next(iout))
            if not (1 <= a < C + i) or not (1 <= b < C + i) or a == b:
                fail("bad gate %d reference" % i)
            wire_val[C + i] = wire_val[a] ^ wire_val[b]
    except Exception:
        fail("truncated/bad gate list")

    total_wires = C + G
    try:
        for r in range(R):
            w = int(next(iout))
            if not (1 <= w <= total_wires):
                fail("bad output wire index for row %d" % (r + 1))
            if wire_val[w] != rows[r]:
                fail("row %d mismatch" % (r + 1))
    except Exception:
        fail("truncated/bad output-row list")

    # no trailing garbage tokens allowed
    leftover = list(iout)
    if leftover:
        fail("trailing tokens in output")

    F = G
    B = sum(bin(mask).count("1") - 1 for mask in rows)
    if B <= 0:
        fail("degenerate baseline")

    sc = min(1000.0, 100.0 * B / max(1e-9, F))
    ratio = sc / 1000.0
    print("G=%d B=%d Ratio: %.6f" % (G, B, ratio))


if __name__ == "__main__":
    main()
