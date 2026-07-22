# TIER: strong
# Inverse design: instead of forward-tweaking a single global match rate (the greedy
# recipe, which cannot tell a donor's counterfactual gift from their marginal response),
# first estimate every donor's UNMATCHED optimum g0_i = clip(a_i-1, 0, w_i). Then pick a
# no-match "shield" tier [0, t1] set at a high quantile of the g0 distribution -- this is
# the target-gifts-first move: t1 marks the region we refuse to subsidize because donors
# would give it anyway. Only above t1 does a positive rate apply, and that rate is BACKED
# OUT (via bisection on the true best-response fixed point, not a naive forward guess) to
# spend exactly the budget on the segmented schedule. A small grid over t1 (several
# quantiles) explores the shield-vs-incentive trade-off and keeps whichever schedule
# raises the most total funds -- this is what escapes the crowd-out fixed point that traps
# a flat, un-segmented rate.
import sys, math


def donor_best_response(a, w, bps, rates):
    K = len(rates)
    Mcum = [0.0] * K
    for k in range(1, K):
        seg_start = 0.0 if k == 1 else bps[k - 2]
        seg_len = bps[k - 1] - seg_start
        Mcum[k] = Mcum[k - 1] + rates[k - 1] * seg_len

    best_g, best_u = 0.0, None
    for k in range(1, K + 1):
        Lk = 0.0 if k == 1 else bps[k - 2]
        Uk = bps[k - 1] if k <= K - 1 else float("inf")
        Uk = min(Uk, w)
        if Lk > w + 1e-9 or Uk < Lk - 1e-9:
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


def payout_and_F(donors, bps, rates):
    total_pay, total_F = 0.0, 0.0
    for a, w in donors:
        g, Mcum = donor_best_response(a, w, bps, rates)
        pay = match_paid(g, bps, rates, Mcum)
        total_pay += pay
        total_F += g + pay
    return total_pay, total_F


def main():
    toks = sys.stdin.read().split()
    p = 0
    N = int(toks[p]); p += 1
    K_MAX = int(toks[p]); p += 1
    R_MAX = float(toks[p]); p += 1
    B = float(toks[p]); p += 1
    donors = []
    for _ in range(N):
        a = float(toks[p]); p += 1
        w = float(toks[p]); p += 1
        donors.append((a, w))

    g0 = []
    for a, w in donors:
        v = a - 1.0
        if v < 0.0: v = 0.0
        if v > w: v = w
        g0.append(v)
    g0.sort()
    n = len(g0)

    best_F, best_t1, best_m2 = -1.0, None, None
    for q in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        idx = min(n - 1, int(q * n))
        t1 = max(0.01, g0[idx])
        lo, hi = 0.0, R_MAX
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            pay, _ = payout_and_F(donors, [t1], [0.0, mid])
            if pay <= B:
                lo = mid
            else:
                hi = mid
        m2 = lo
        pay, F = payout_and_F(donors, [t1], [0.0, m2])
        if F > best_F:
            best_F, best_t1, best_m2 = F, t1, m2

    if best_t1 is None:
        # degenerate fallback (shouldn't happen): no match at all
        print("1")
        print("0.0")
        return

    m2 = best_m2 * (1.0 - 1e-9)  # tiny shrink to stay strictly within budget under FP
    print("2")
    print("%.10f" % best_t1)
    print("0.0 %.10f" % m2)


if __name__ == "__main__":
    main()
