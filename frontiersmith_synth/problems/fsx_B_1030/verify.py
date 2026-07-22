#!/usr/bin/env python3
# Deterministic checker for cherrypick-rate-board (format C, MAXIMIZE net
# harvest = value of picked fields minus wages paid).
# CLI: python3 verify.py <in> <out> <ans>   (ans ignored).
# Prints "... Ratio: <r>" with r in [0,1]; any feasibility breach -> Ratio: 0.0.
import sys


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def ceil_div(a, b):
    return -(-a // b)


def cost(c_unit, d, s):
    return ceil_div(c_unit * d, s)


def simulate(N, M, fields, workers, rates, c_unit):
    """Replay the arrival process: worker j (1-indexed) sees the fields still
    available (not yet taken, deadline not yet passed) and takes the one
    maximizing rate-cost margin, provided that margin clears their reservation
    wage; ties broken by lowest field index. Returns (H, W) = harvested value,
    wages paid."""
    avail = set(range(N))
    H = 0
    W = 0
    for j in range(1, M + 1):
        s, w = workers[j - 1]
        if avail:
            expired = [i for i in avail if fields[i][2] < j]
            for i in expired:
                avail.discard(i)
        if not avail:
            continue
        best_i, best_margin = -1, None
        for i in avail:
            v, d, dl = fields[i]
            m = rates[i] - cost(c_unit, d, s)
            if best_margin is None or m > best_margin or (m == best_margin and i < best_i):
                best_margin = m
                best_i = i
        if best_margin is not None and best_margin >= w:
            v, d, dl = fields[best_i]
            H += v
            W += rates[best_i]
            avail.discard(best_i)
    return H, W


def main():
    # ---- instance ------------------------------------------------------
    try:
        it = open(sys.argv[1]).read().split()
    except Exception:
        fail("bad instance")
    p = 0
    N = int(it[p]); M = int(it[p + 1]); C_UNIT = int(it[p + 2]); RMAX = int(it[p + 3]); p += 4
    fields = []
    for _ in range(N):
        v = int(it[p]); d = int(it[p + 1]); dl = int(it[p + 2]); p += 3
        fields.append((v, d, dl))
    workers = []
    for _ in range(M):
        s = int(it[p]); w = int(it[p + 1]); p += 2
        workers.append((s, w))

    # ---- participant output ---------------------------------------------
    try:
        ot = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if len(ot) != N:
        fail("expected exactly %d rate tokens, got %d" % (N, len(ot)))
    rates = []
    for k, tok in enumerate(ot):
        try:
            r = int(tok)
        except Exception:
            fail("rate %d (%r) is not an integer" % (k, tok))
        if r < 0 or r > RMAX:
            fail("rate %d = %d out of range [0,%d]" % (k, r, RMAX))
        rates.append(r)

    H, W = simulate(N, M, fields, workers, rates, C_UNIT)
    F_obj = H - W

    # ---- internal baseline: post the SAME flat rate on every field --------
    R_FLAT = round(0.5 * RMAX)
    Hb, Wb = simulate(N, M, fields, workers, [R_FLAT] * N, C_UNIT)
    B = Hb - Wb
    if B <= 0:
        B = 1e-9

    sc = max(0.0, min(1000.0, 100.0 * F_obj / max(1e-9, B)))
    print("H=%d W=%d F=%d B=%.1f Ratio: %.6f" % (H, W, F_obj, B, sc / 1000.0))


if __name__ == "__main__":
    main()
