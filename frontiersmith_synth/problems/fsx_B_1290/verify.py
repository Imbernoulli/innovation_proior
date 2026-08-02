#!/usr/bin/env python3
# Deterministic checker for payment-routing-cost (format C, MINIMIZE cost per
# successful payment).  CLI: python3 verify.py <in> <out> <ans>  (ans ignored).
# Prints "... Ratio: <r>" with r in [0,1]; any feasibility breach -> Ratio: 0.0.
import sys


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def fee_of(fixedFee, pctBps, r, b, amt):
    return fixedFee[r][b] + pctBps[r][b] * amt[b] / 10000.0


def cell_ratio(seq, s, b, auth, fixedFee, pctBps, retry, amt, failpen_bps):
    """Expected cost per successful payment for cascade `seq` (list of rail
    indices, distinct, length>=1) on issuer segment s / ticket bucket b."""
    failpen = failpen_bps * amt[b] / 10000.0
    cost = 0.0
    reach = 1.0
    for i, r in enumerate(seq, start=1):
        p = auth[s][r]
        q = 1.0 - p
        attempt_cost = fee_of(fixedFee, pctBps, r, b, amt) + (retry[r] if i > 1 else 0.0)
        cost += reach * attempt_cost
        reach *= q
    cost += reach * failpen           # tail: every attempt in the cascade declined
    succ = 1.0 - reach
    return cost / max(succ, 1e-9)


def main():
    # ---- instance ------------------------------------------------------
    try:
        it = open(sys.argv[1]).read().split()
    except Exception:
        fail("bad instance")
    p = 0
    R = int(it[p]); S = int(it[p + 1]); B = int(it[p + 2]); K = int(it[p + 3]); FAILPEN_BPS = int(it[p + 4]); p += 5
    amt = [int(it[p + i]) for i in range(B)]; p += B
    fixedFee = [[0] * B for _ in range(R)]
    pctBps = [[0] * B for _ in range(R)]
    retry = [0] * R
    for r in range(R):
        for b in range(B):
            fixedFee[r][b] = int(it[p]); pctBps[r][b] = int(it[p + 1]); p += 2
        retry[r] = int(it[p]); p += 1
    auth = [[0.0] * R for _ in range(S)]
    for s in range(S):
        for r in range(R):
            auth[s][r] = float(it[p]); p += 1
    vol = [[0] * B for _ in range(S)]
    for s in range(S):
        for b in range(B):
            vol[s][b] = int(it[p]); p += 1

    # ---- participant output ---------------------------------------------
    try:
        raw = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")
    if not raw:
        fail("empty output")

    n_cells = S * B
    policies = []  # policies[s*B+b] = list of rail indices
    idx = 0
    for cellno in range(n_cells):
        if idx >= len(raw):
            fail("truncated output at cell %d" % cellno)
        try:
            L = int(raw[idx])
        except Exception:
            fail("non-integer cascade length at cell %d" % cellno)
        idx += 1
        if L < 1 or L > K or L > R:
            fail("cascade length %d out of range [1,%d] at cell %d" % (L, min(K, R), cellno))
        if idx + L > len(raw):
            fail("truncated cascade at cell %d" % cellno)
        seq = []
        seen = set()
        for _ in range(L):
            try:
                r = int(raw[idx])
            except Exception:
                fail("non-integer rail id at cell %d" % cellno)
            idx += 1
            if r < 0 or r >= R:
                fail("rail id %d out of range at cell %d" % (r, cellno))
            if r in seen:
                fail("duplicate rail %d in cascade at cell %d" % (r, cellno))
            seen.add(r)
            seq.append(r)
        policies.append(seq)
    if idx != len(raw):
        fail("trailing garbage after expected %d cells" % n_cells)

    # ---- participant objective: volume-weighted mean cost-per-success ----
    tot_vol = 0
    tot_cost = 0.0
    for s in range(S):
        for b in range(B):
            seq = policies[s * B + b]
            r = cell_ratio(seq, s, b, auth, fixedFee, pctBps, retry, amt, FAILPEN_BPS)
            v = vol[s][b]
            tot_cost += v * r
            tot_vol += v
    if tot_vol <= 0:
        fail("degenerate instance (zero volume)")
    F_val = tot_cost / tot_vol

    # ---- internal baseline: single globally-cheapest-average-fee rail, ----
    # ---- attempted once, no cascade, used for every segment/bucket ------
    avg_fee = []
    for r in range(R):
        avg_fee.append(sum(fee_of(fixedFee, pctBps, r, b, amt) for b in range(B)) / B)
    best_r = min(range(R), key=lambda r: (avg_fee[r], r))
    base_cost = 0.0
    for s in range(S):
        for b in range(B):
            r = cell_ratio([best_r], s, b, auth, fixedFee, pctBps, retry, amt, FAILPEN_BPS)
            base_cost += vol[s][b] * r
    B_val = base_cost / tot_vol
    if B_val <= 0:
        B_val = 1e-9

    sc = min(1000.0, 100.0 * B_val / max(1e-9, F_val))
    print("F=%.4f B=%.4f Ratio: %.6f" % (F_val, B_val, sc / 1000.0))


if __name__ == "__main__":
    main()
