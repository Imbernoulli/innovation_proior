#!/usr/bin/env python3
# Deterministic checker for credit-limit-assign (format C, maximize portfolio profit
# under a loss cap). CLI: python3 verify.py <in> <out> <ans>  (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1] on its own final line, and exits 0.
import sys, math

BASE_TIER = 2   # must match gen.py's BASE_TIER (the checker's own trivial baseline)


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def profit_loss(L, m, d_bps, u_bps, apr_bps, sev_bps):
    if m == 0:
        return 0, 0
    balance = L[m] * u_bps // 10000
    revenue = balance * apr_bps // 10000 * (10000 - d_bps) // 10000
    loss = balance * d_bps // 10000 * sev_bps // 10000
    return revenue - loss, loss


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
        p = 0
        N = int(itoks[p]); p += 1
        M = int(itoks[p]); p += 1
        L = [0] * (M + 1)
        for m in range(1, M + 1):
            L[m] = int(itoks[p]); p += 1
        apr_bps = int(itoks[p]); p += 1
        sev_bps = int(itoks[p]); p += 1
        cap = int(itoks[p]); p += 1
        borrowers = []   # (score, d[1..M], u[1..M])
        for _ in range(N):
            score = int(itoks[p]); p += 1
            d = [0] * (M + 1)
            u = [0] * (M + 1)
            for m in range(1, M + 1):
                d[m] = int(itoks[p]); p += 1
                u[m] = int(itoks[p]); p += 1
            borrowers.append((score, d, u))
        if p != len(itoks):
            fail("trailing/garbled instance")
        if N < 1 or M < 1 or cap < 0:
            fail("bad instance ranges")
    except Exception:
        fail("bad instance")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if len(otoks) != N:
        fail("expected %d tokens, got %d" % (N, len(otoks)))

    choices = []
    for k in range(N):
        tok = otoks[k]
        try:
            # reject non-integer / non-finite tokens (nan, inf, "1.5", etc.)
            if any(c not in "-0123456789" for c in tok):
                raise ValueError
            m = int(tok)
        except Exception:
            fail("token %d (%r) is not an integer tier index" % (k, tok))
        if m < 0 or m > M:
            fail("tier %d out of range at applicant %d" % (m, k))
        choices.append(m)

    total_profit = 0
    total_loss = 0
    for i in range(N):
        score, d, u = borrowers[i]
        m = choices[i]
        prof, loss = profit_loss(L, m, d[m] if m else 0, u[m] if m else 0, apr_bps, sev_bps)
        total_profit += prof
        total_loss += loss

    if total_loss > cap:
        fail("portfolio loss %d exceeds cap %d" % (total_loss, cap))

    F = total_profit

    # internal trivial baseline B: process applicants in INPUT ORDER, give each
    # BASE_TIER as long as it doesn't break the cap, else decline (tier 0).
    b_loss = 0
    B = 0
    for i in range(N):
        score, d, u = borrowers[i]
        prof, loss = profit_loss(L, BASE_TIER, d[BASE_TIER], u[BASE_TIER], apr_bps, sev_bps)
        if b_loss + loss <= cap:
            b_loss += loss
            B += prof

    if F <= 0 or B <= 0:
        # a non-positive portfolio (or a degenerate baseline) never earns credit
        print("F=%d B=%d Ratio: 0.0" % (F, B))
        return

    sc = min(1000.0, 100.0 * F / max(1e-9, float(B)))
    ratio = sc / 1000.0
    if not math.isfinite(ratio):
        fail("non-finite ratio")
    print("F=%d B=%d Ratio: %.6f" % (F, B, ratio))


if __name__ == "__main__":
    main()
