import sys, math
from fractions import Fraction as Fr

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def depth(idx):
    return (idx + 1).bit_length() - 1

def main():
    try:
        inp = open(sys.argv[1]).read().split()
    except Exception:
        fail("cannot read input")

    it = iter(inp)
    try:
        T = int(next(it))
        cost = [int(next(it)) for _ in range(T)]
        M = 2 ** (T + 1) - 1
        S = [int(next(it)) for _ in range(M)]
        NLEAF = 2 ** T
        L = [int(next(it)) for _ in range(NLEAF)]
    except Exception:
        fail("malformed input")

    # ---- internal baseline B: settle at the very first opportunity (round 0) ----
    B = float(S[0]) if S[0] > 0 else 1e-9

    # ---- parse participant output: "<M>\n<M-char policy string over {S,C}>" ----
    try:
        out_txt = open(sys.argv[2]).read()
    except Exception:
        fail("cannot read output")
    toks = out_txt.split()
    if len(toks) < 1:
        fail("empty output")

    try:
        fm = float(toks[0])
    except Exception:
        fail("bad node-count token")
    if not math.isfinite(fm):
        fail("non-finite node count")
    if fm != int(fm):
        fail("node count not an integer")
    m = int(fm)
    if m != M:
        fail("node count %d != expected %d" % (m, M))

    if len(toks) != 1 + M:
        fail("expected %d policy tokens, got %d" % (M, len(toks) - 1))

    policy = toks[1:1 + M]
    for tok in policy:
        if tok not in ("S", "C"):
            fail("bad decision token %r (must be S or C)" % tok)

    # ---- accrued cost per node (heap order), then backward-evaluate the
    #      submitted policy exactly (Fraction arithmetic -- no rounding) ----
    accrued = [Fr(0)] * M
    for idx in range(1, M):
        parent = (idx - 1) // 2
        accrued[idx] = accrued[parent] + cost[depth(parent)]

    value = [None] * M
    for idx in reversed(range(M)):
        d = depth(idx)
        if policy[idx] == "S":
            value[idx] = Fr(S[idx]) - accrued[idx]
        else:  # 'C'
            if d == T:
                leaf_j = idx - (2 ** T - 1)
                value[idx] = Fr(L[leaf_j]) - accrued[idx]
            else:
                left = value[2 * idx + 1]
                right = value[2 * idx + 2]
                value[idx] = Fr(1, 2) * (left + right)

    F = float(value[0])
    sc = min(1000.0, 100.0 * max(0.0, F) / max(1e-9, B))
    print("F=%.4f B=%.4f Ratio: %.6f" % (F, B, sc / 1000.0))

if __name__ == "__main__":
    main()
