# TIER: greedy
# The "obvious" textbook approach: ONE flat match rate applied from the very first
# dollar (K=1), calibrated by binary search to spend the whole budget. This is the
# forward-tweak-a-single-lever recipe -- it never distinguishes a donor's counterfactual
# giving from their marginal response, so on a bimodal donor pool it burns most of the
# budget matching gifts that would have happened anyway.
import sys, math


def donor_best_response(a, w, rate):
    # K=1 schedule: M(g) = rate*g everywhere.
    m = rate
    candidates = [0.0, w]
    g_star = a - 1.0 / (1.0 + m)
    if 0.0 - 1e-9 <= g_star <= w + 1e-9:
        candidates.append(min(max(g_star, 0.0), w))
    best_g, best_u = 0.0, None
    for g in candidates:
        val = 1.0 + g + m * g
        if val <= 0:
            continue
        u = a * math.log(val) - g
        if best_u is None or u > best_u + 1e-12:
            best_u = u
            best_g = g
    return best_g


def payout_and_F(donors, rate):
    total_pay = 0.0
    total_F = 0.0
    for a, w in donors:
        g = donor_best_response(a, w, rate)
        pay = rate * g
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

    lo, hi = 0.0, R_MAX
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pay, _ = payout_and_F(donors, mid)
        if pay <= B:
            lo = mid
        else:
            hi = mid
    rate = lo * (1.0 - 1e-9)  # tiny shrink to stay strictly within budget under FP

    print("1")
    print("%.10f" % rate)


if __name__ == "__main__":
    main()
