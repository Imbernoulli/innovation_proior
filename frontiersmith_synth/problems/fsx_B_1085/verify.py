#!/usr/bin/env python3
# Deterministic checker for seek-window-arrangement (format C, MINIMISE total
# pickup-arm travel).  CLI: python3 verify.py <in> <out> <ans>  (ans ignored).
# Prints "... Ratio: <r>" with r in [0,1]; any feasibility breach -> Ratio: 0.0.
import sys


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def simulate(N, T, cap, w, Q, trk):
    """Fixed bounded-reorder-window pickup-arm rule. Buffer holds the oldest
    (up to) w not-yet-played cue-sheet indices; always play whichever buffered
    cue sits on the nearest groove to the current arm position (ties -> the
    earliest cue index). Returns total groove-to-groove travel."""
    M = len(Q)
    pos = 0
    buf = []
    while pos < M and len(buf) < w:
        buf.append(pos)
        pos += 1
    arm = 0
    cost = 0
    while buf:
        best_i = 0
        best_key = None
        for i, qi in enumerate(buf):
            diff = abs(trk[Q[qi]] - arm)
            key = (diff, qi)
            if best_key is None or key < best_key:
                best_key = key
                best_i = i
        qi = buf.pop(best_i)
        cost += abs(trk[Q[qi]] - arm)
        arm = trk[Q[qi]]
        if pos < M:
            buf.append(pos)
            pos += 1
    return cost


def main():
    # ---- instance ------------------------------------------------------
    try:
        it = open(sys.argv[1]).read().split()
    except Exception:
        fail("bad instance")
    p = 0
    N = int(it[p]); T = int(it[p + 1]); cap = int(it[p + 2]); w = int(it[p + 3]); M = int(it[p + 4])
    p += 5
    Q = [int(it[p + k]) for k in range(M)]
    p += M

    # ---- participant output ---------------------------------------------
    try:
        ot = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if len(ot) < N:
        fail("truncated output: need %d track ids, got %d" % (N, len(ot)))
    if len(ot) > N:
        fail("too many tokens: expected exactly %d" % N)

    trk = []
    for k in range(N):
        tok = ot[k]
        try:
            v = float(tok)
        except Exception:
            fail("non-numeric token at position %d" % k)
        if v != v or v in (float("inf"), float("-inf")):
            fail("non-finite value at position %d" % k)
        iv = int(tok) if tok.lstrip("-").isdigit() else None
        if iv is None or float(iv) != v:
            fail("track id must be an integer at position %d" % k)
        if iv < 0 or iv >= T:
            fail("track id %d out of range [0,%d) at position %d" % (iv, T, k))
        trk.append(iv)

    counts = [0] * T
    for v in trk:
        counts[v] += 1
    for tr, c in enumerate(counts):
        if c > cap:
            fail("groove %d holds %d cells > capacity %d" % (tr, c, cap))

    F_obj = simulate(N, T, cap, w, Q, trk)

    # ---- internal baseline: press cells in raw id order, packed by cap ----
    trk_base = [min(T - 1, i // cap) for i in range(N)]
    B = simulate(N, T, cap, w, Q, trk_base)
    if B <= 0:
        B = 1

    sc = min(1000.0, 100.0 * B / max(1, F_obj))
    print("F_obj=%d B=%d Ratio: %.6f" % (F_obj, B, sc / 1000.0))


if __name__ == "__main__":
    main()
