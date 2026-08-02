import sys

# Format D checker -- block-partition / bit-width codec with a branch-misprediction tax.
#
#   Input  <in>:  N H C
#                 A_1 ... A_N            (0 <= A_i < 2**30)
#   Output <out>: M
#                 len_1 width_1
#                 ...
#                 len_M width_M
#
# Block k spans the next len_k elements; base_k = min of its elements,
# d_k = max of its elements - base_k. Feasible iff d_k <= 2**width_k - 1
# (and d_k == 0 when width_k == 0) -- this IS the exact-reconstruction check:
# a feasible block decodes bit-for-bit back to its original elements via
# value = base_k + stored_delta, so feasibility == exact equivalence here.
#
# Objective (minimize):
#   cost = sum_k (H + len_k * width_k)
#          + C * (# adjacent block pairs whose width differs)
# Baseline B = cost of the single block covering the whole stream at its own
# minimal feasible width (checker's own trivial construction, no transitions).
#   Ratio = min(1000, 100 * B / cost) / 1000

WMAX = 30
VMAX = (1 << 30) - 1
MAX_TOKEN_DIGITS = 15  # guards against bignum-DoS garbage tokens


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def parse_int(tok, lo=None, hi=None):
    if not tok:
        raise ValueError("empty token")
    t = tok[1:] if tok[0] in "+-" else tok
    if not t.isdigit() or len(t) > MAX_TOKEN_DIGITS:
        raise ValueError("not a small plain integer: %r" % tok)
    v = int(tok)
    if lo is not None and v < lo:
        raise ValueError("below range")
    if hi is not None and v > hi:
        raise ValueError("above range")
    return v


def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    it = iter(inp)
    try:
        N = int(next(it)); H = int(next(it)); C = int(next(it))
        A = [int(next(it)) for _ in range(N)]
    except Exception:
        fail("bad instance (should never happen)")

    # ---- baseline B: single block, whole stream, its own minimal width ----
    base_all = min(A)
    d_all = max(A) - base_all
    w_all = d_all.bit_length()
    B = H + N * w_all
    if B <= 0:
        fail("degenerate baseline")

    # ---- parse participant output ----
    if not out:
        fail("empty output")
    try:
        M = parse_int(out[0], lo=1, hi=N)
    except Exception as e:
        fail("bad M (%s)" % e)

    need = 1 + 2 * M
    if len(out) != need:
        fail("wrong token count (got %d, need %d)" % (len(out), need))

    blocks = []
    try:
        p = 1
        for _ in range(M):
            ln = parse_int(out[p], lo=1, hi=N); p += 1
            wd = parse_int(out[p], lo=0, hi=WMAX); p += 1
            blocks.append((ln, wd))
    except Exception as e:
        fail("bad block token (%s)" % e)

    total_len = sum(ln for ln, _ in blocks)
    if total_len != N:
        fail("block lengths sum to %d, need %d" % (total_len, N))

    # ---- feasibility (== exact reconstruction) + cost ----
    cost = 0
    prev_w = None
    idx = 0
    for (ln, wd) in blocks:
        seg = A[idx:idx + ln]
        base = min(seg)
        d = max(seg) - base
        cap = (1 << wd) - 1 if wd > 0 else 0
        if d > cap:
            fail("block at offset %d (len %d, width %d) cannot represent delta %d (cap %d)"
                 % (idx, ln, wd, d, cap))
        cost += H + ln * wd
        if prev_w is not None and wd != prev_w:
            cost += C
        prev_w = wd
        idx += ln

    ratio = min(1000.0, 100.0 * B / max(1e-9, cost)) / 1000.0
    print("B=%d cost=%d M=%d Ratio: %.6f" % (B, cost, M, ratio))


if __name__ == "__main__":
    main()
