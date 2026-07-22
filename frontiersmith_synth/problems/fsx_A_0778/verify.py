#!/usr/bin/env python3
# Deterministic checker for "Museum Match Curve" (format C, maximize total funds raised).
# CLI: python3 verify.py <in> <out> <ans>   (ans is ignored)
# Prints "... Ratio: <r>" with r in [0,1].
import sys, math

TOL = 1e-6
BASE_FRACTION = 0.4  # internal baseline = BASE_FRACTION * (no-match counterfactual total)


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def is_finite(x):
    return isinstance(x, float) and math.isfinite(x)


def donor_best_response(a, w, bps, rates):
    """Exact best response: donor i picks g in [0, w] maximizing
    U(g) = a * ln(1 + g + M(g)) - g, where M is the piecewise-linear schedule
    (breakpoints bps, rates 'rates', M(0)=0). Because the marginal match rate can
    JUMP UP at a kink, U need not be globally concave across segments, so we
    evaluate every segment's local optimum (interior stationary point + both
    endpoints) and take the global best -- this is the same finite candidate set
    for any caller (checker and reference solutions alike)."""
    K = len(rates)
    # cumulative match paid at the START of each segment (Mcum[k-1] = M at L_k)
    Mcum = [0.0] * K
    for k in range(1, K):
        seg_start = 0.0 if k == 1 else bps[k - 2]
        seg_len = bps[k - 1] - seg_start
        Mcum[k] = Mcum[k - 1] + rates[k - 1] * seg_len

    best_g = 0.0
    best_u = None
    for k in range(1, K + 1):
        Lk = 0.0 if k == 1 else bps[k - 2]
        Uk = bps[k - 1] if k <= K - 1 else float("inf")
        Uk = min(Uk, w)
        if Lk > w + 1e-9:
            continue
        if Uk < Lk - 1e-9:
            continue
        Uk = max(Uk, Lk)
        m_k = rates[k - 1]
        C_k = 1.0 + Mcum[k - 1] - m_k * Lk
        candidates = [Lk, Uk]
        g_star = a - C_k / (1.0 + m_k)
        if Lk - 1e-9 <= g_star <= Uk + 1e-9:
            candidates.append(min(max(g_star, Lk), Uk))
        for g in candidates:
            M_g = Mcum[k - 1] + m_k * (g - Lk)
            val = 1.0 + g + M_g
            if val <= 0:
                continue
            u = a * math.log(val) - g
            if best_u is None or u > best_u + 1e-12:
                best_u = u
                best_g = g
    return best_g, Mcum


def match_paid(g, bps, rates, Mcum):
    K = len(rates)
    for k in range(1, K + 1):
        Lk = 0.0 if k == 1 else bps[k - 2]
        Uk = bps[k - 1] if k <= K - 1 else float("inf")
        if g <= Uk + 1e-9:
            return Mcum[k - 1] + rates[k - 1] * (g - Lk)
    Lk = 0.0 if K == 1 else bps[K - 2]
    return Mcum[K - 1] + rates[K - 1] * (g - Lk)


def main():
    try:
        itoks = open(sys.argv[1]).read().split()
        p = 0
        N = int(itoks[p]); p += 1
        K_MAX = int(itoks[p]); p += 1
        R_MAX = float(itoks[p]); p += 1
        B = float(itoks[p]); p += 1
        donors = []
        for _ in range(N):
            a = float(itoks[p]); p += 1
            w = float(itoks[p]); p += 1
            donors.append((a, w))
    except Exception:
        fail("bad instance")

    try:
        otoks = open(sys.argv[2]).read().split()
    except Exception:
        fail("no output")

    if not otoks:
        fail("empty output")

    try:
        K = int(float(otoks[0]))
    except Exception:
        fail("bad K")

    if K < 1 or K > K_MAX:
        fail("K out of range")

    need = 1 + (K - 1) + K
    if len(otoks) < need:
        fail("truncated schedule")

    try:
        bps = [float(t) for t in otoks[1:1 + (K - 1)]]
        rates = [float(t) for t in otoks[1 + (K - 1):1 + (K - 1) + K]]
    except Exception:
        fail("bad schedule tokens")

    for v in bps + rates:
        if not is_finite(v):
            fail("non-finite schedule value")

    for i, t in enumerate(bps):
        if t <= TOL:
            fail("breakpoint %d not positive" % i)
        if i > 0 and t <= bps[i - 1] + TOL:
            fail("breakpoints not strictly increasing")

    for i, m in enumerate(rates):
        if m < -TOL or m > R_MAX + TOL:
            fail("rate %d out of [0,R_MAX]" % i)
    rates = [max(0.0, m) for m in rates]

    # ---- donor best responses under the submitted schedule ----
    total_payout = 0.0
    total_F = 0.0
    for a, w in donors:
        g_star, Mcum = donor_best_response(a, w, bps, rates)
        if not (math.isfinite(g_star) and -TOL <= g_star <= w + TOL):
            fail("donor best response out of range")
        g_star = min(max(g_star, 0.0), w)
        pay = match_paid(g_star, bps, rates, Mcum)
        if not math.isfinite(pay) or pay < -TOL:
            fail("non-finite/negative match payout")
        pay = max(0.0, pay)
        total_payout += pay
        total_F += g_star + pay

    budget_tol = 1e-6 * max(1.0, B)
    if total_payout > B + budget_tol:
        fail("budget exceeded: payout=%.6f > B=%.6f" % (total_payout, B))

    F = total_F

    # internal baseline: BASE_FRACTION of what the museum would raise with NO matching
    # program at all (each donor's own unmatched log-utility optimum a_i - 1, capped by
    # their wealth w_i) -- a deliberately modest, checker-built reference.
    sumg0 = 0.0
    for a, w in donors:
        g0 = a - 1.0
        if g0 < 0.0:
            g0 = 0.0
        if g0 > w:
            g0 = w
        sumg0 += g0
    BASE = BASE_FRACTION * sumg0

    sc = min(1000.0, 100.0 * F / max(1e-9, BASE))
    print("F=%.6f payout=%.6f B=%.6f BASE=%.6f Ratio: %.6f" % (F, total_payout, B, BASE, sc / 1000.0))


if __name__ == "__main__":
    main()
