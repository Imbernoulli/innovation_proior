# TIER: strong
# Insight: on a "safe" day (not a month's last day) b[t+1] = floor(b[t]*(1+r))
# exactly pins r into the half-open interval
#   [ (b[t+1]-b[t])/b[t] ,  (b[t+1]-b[t]+1)/b[t] ).
# Intersecting this interval over every safe day of every pooled account
# narrows it to (almost always) a single point; the SIMPLEST fraction inside
# the narrowed interval (Stern-Brocot / continued-fraction search) is the
# exact hidden rate. The "rounding residue" is not noise -- it IS r.
# With r exact, every month-end shortfall is either exactly 0 (no fee) or
# exactly the true fee (fee triggered), so fee and the poverty line theta
# read off directly and reproduce the rule exactly forever.
import sys
import math
from fractions import Fraction

L = 30


def simplest_in_interval(lo: Fraction, hi: Fraction) -> Fraction:
    """Simplest (smallest-denominator) fraction in [lo, hi], via a
    Stern-Brocot / continued-fraction descent."""
    if lo > hi:
        lo, hi = hi, lo

    def sb(lo: Fraction, hi: Fraction) -> Fraction:
        a0 = math.floor(lo)
        if a0 == math.floor(hi) and hi != a0:
            if lo == a0:
                return Fraction(a0)
            return a0 + 1 / sb(1 / (hi - a0), 1 / (lo - a0))
        else:
            c = math.ceil(lo)
            return Fraction(c) if c <= hi else Fraction(a0 + 1)

    return sb(lo, hi)


def main():
    data = sys.stdin.read().split()
    idx = 0
    test_id = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1
    T = int(data[idx]); idx += 1
    Lm = int(data[idx]); idx += 1
    accounts = []
    for _ in range(K):
        bal = [int(x) for x in data[idx:idx + T + 1]]
        idx += T + 1
        accounts.append(bal)

    # 1) exact r via interval intersection over all safe days
    lo, hi = Fraction(0), Fraction(1)
    for bal in accounts:
        for t in range(T):
            if t % Lm != Lm - 1:
                b, bn = bal[t], bal[t + 1]
                if b <= 0:
                    continue
                rl = Fraction(bn - b, b)
                rh = Fraction(bn - b + 1, b)
                if rl > lo:
                    lo = rl
                if rh < hi:
                    hi = rh
    if lo >= hi:
        r = lo  # degenerate: interval collapsed to (at most) a point
    else:
        r = simplest_in_interval(lo, hi)
    p, q = r.numerator, r.denominator
    if q == 0:
        p, q = 0, 1

    # 2) exact fee + theta from month-end shortfalls under the exact rate
    fee_candidates = []
    fee_month_starts = []
    nofee_month_starts = []
    for bal in accounts:
        nmonths = T // Lm
        for m in range(nmonths):
            t_end = m * Lm + (Lm - 1)
            if t_end + 1 > T:
                continue
            b_last, b_next = bal[t_end], bal[t_end + 1]
            grown_exact = (b_last * (q + p)) // q
            delta = grown_exact - b_next
            month_start = bal[m * Lm]
            if delta > 0:
                fee_candidates.append(delta)
                fee_month_starts.append(month_start)
            else:
                nofee_month_starts.append(month_start)

    if fee_candidates:
        # mode (should all agree exactly once r is exact)
        counts = {}
        for v in fee_candidates:
            counts[v] = counts.get(v, 0) + 1
        fee = max(counts.items(), key=lambda kv: kv[1])[0]
    else:
        fee = 0

    if fee_month_starts and nofee_month_starts:
        theta = max(fee_month_starts) + 1
    elif fee_month_starts:
        theta = max(fee_month_starts) + 1
    elif nofee_month_starts:
        theta = min(nofee_month_starts)
    else:
        theta = 0

    print(f"{p} {q} {fee} {theta}")


if __name__ == "__main__":
    main()
