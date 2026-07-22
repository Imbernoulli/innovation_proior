# TIER: strong
"""Insight: once the ladder is fixed the season decomposes into N independent
per-night subproblems (each with only ~headroom legal sale caps), so evaluate every
legal cap exactly against the printed data. The hard part is the ladder: build several
candidates from FARE-WEIGHTED quantiles of the threshold distribution (a night's
revenue-at-risk, not its raw passenger count, decides how much it shapes the shared
ladder) plus a peak-only candidate, evaluate each exactly via the per-night optimum,
and keep the best (ladder, cap-vector) pair. This is the binding-quantile-not-mean
insight: caps are only safe on nights whose OWN threshold quantile the chosen ladder
actually reaches."""
import sys

INF = float("inf")


def make_valid_ladder(vals, lo, hi):
    v = sorted(max(lo, min(hi, int(round(x)))) for x in vals)
    for i in range(1, 5):
        if v[i] <= v[i - 1]:
            v[i] = v[i - 1] + 1
    if v[4] > hi:
        v[4] = hi
        for i in range(3, -1, -1):
            if v[i] >= v[i + 1]:
                v[i] = v[i + 1] - 1
    return v


def weighted_percentile(pairs_sorted, cum_weight, total_weight, p):
    """pairs_sorted: list of (threshold, weight) sorted by threshold.
    cum_weight: prefix sums of weight aligned with pairs_sorted."""
    target = p / 100.0 * total_weight
    lo, hi = 0, len(pairs_sorted) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cum_weight[mid] >= target:
            hi = mid
        else:
            lo = mid + 1
    return pairs_sorted[lo][0]


def best_sold_for_night(capacity, fare, max_sell, noshow, threshold, ladder):
    """Exact per-night optimum given a fixed ladder: sold ranges over [capacity, max_sell]
    (selling below capacity is never beneficial -- at sold=capacity, shows<=capacity
    always, i.e. zero risk, so undershooting capacity only loses guaranteed revenue)."""
    top = ladder[-1]

    def cost_of(j):
        th = threshold[j]
        if th > top:
            return INF
        for step in ladder:
            if step >= th:
                return step
        return INF

    best_s, best_val = capacity, capacity * fare
    for s in range(capacity, max_sell + 1):
        shown = [j for j in range(s) if noshow[j] == 0]
        val = s * fare
        shows = len(shown)
        if shows > capacity:
            overflow = shows - capacity
            ranked = sorted(shown, key=lambda j: (cost_of(j), j))
            bumped = ranked[:overflow]
            for j in bumped:
                c = cost_of(j)
                if c == INF:
                    val -= PENALTY[0]
                else:
                    val -= c
        if val > best_val:
            best_val = val
            best_s = s
    return best_s, best_val


PENALTY = [900]


def main():
    toks = sys.stdin.read().split()
    pos = 0
    n = int(toks[pos]); pos += 1
    ladder_lo, ladder_hi, penalty = int(toks[pos]), int(toks[pos + 1]), int(toks[pos + 2])
    pos += 3
    PENALTY[0] = penalty
    nights = []
    for _ in range(n):
        capacity, fare, max_sell = int(toks[pos]), int(toks[pos + 1]), int(toks[pos + 2])
        pos += 3
        noshow = [0] * max_sell
        threshold = [0] * max_sell
        for j in range(max_sell):
            ns, th = int(toks[pos]), int(toks[pos + 1])
            pos += 2
            noshow[j] = ns
            threshold[j] = th
        nights.append((capacity, fare, max_sell, noshow, threshold))

    fares = sorted(f for (c, f, m, ns, th) in nights)
    med_fare = fares[len(fares) // 2] if fares else 0

    # fare-weighted threshold distribution: a passenger's threshold counts proportional
    # to that night's fare (= revenue at risk), so high-fare nights shape the ladder more
    weighted = []
    for (c, f, m, ns, th) in nights:
        for j in range(m):
            weighted.append((th[j], f))
    weighted.sort(key=lambda x: x[0])
    cw = []
    acc = 0
    for _, w in weighted:
        acc += w
        cw.append(acc)
    total_w = acc if weighted else 1

    def wpct(p):
        return weighted_percentile(weighted, cw, total_w, p)

    cand1 = make_valid_ladder([wpct(15), wpct(35), wpct(55), wpct(78), wpct(95)], ladder_lo, ladder_hi)

    # peak-only candidate: quantiles from nights whose fare exceeds the median fare
    peak_thresholds = []
    for (c, f, m, ns, th) in nights:
        if f > med_fare:
            peak_thresholds.extend(th)
    peak_thresholds.sort()

    def ppct(p):
        if not peak_thresholds:
            return ladder_lo
        idx = int(p / 100.0 * (len(peak_thresholds) - 1))
        idx = max(0, min(len(peak_thresholds) - 1, idx))
        return peak_thresholds[idx]

    cand2 = make_valid_ladder([ppct(20), ppct(50), ppct(75), ppct(90), ppct(99)], ladder_lo, ladder_hi)
    cand2b = make_valid_ladder([ppct(5), ppct(30), ppct(60), ppct(85), ppct(100)], ladder_lo, ladder_hi)

    all_thresholds = sorted(th for (c, f, m, ns, th) in nights for th in th)

    def apct(p):
        idx = int(p / 100.0 * (len(all_thresholds) - 1))
        idx = max(0, min(len(all_thresholds) - 1, idx))
        return all_thresholds[idx]

    cand3 = make_valid_ladder([apct(10), apct(30), apct(50), apct(70), apct(90)], ladder_lo, ladder_hi)
    cand4 = make_valid_ladder([apct(30), apct(55), apct(75), apct(90), apct(99)], ladder_lo, ladder_hi)
    cand5 = make_valid_ladder([(a + b) / 2.0 for a, b in zip(cand1, cand2)], ladder_lo, ladder_hi)
    cand6 = make_valid_ladder([(a + b) / 2.0 for a, b in zip(cand1, cand2b)], ladder_lo, ladder_hi)

    candidates = [cand1, cand2, cand2b, cand3, cand4, cand5, cand6]

    best_total = -float("inf")
    best_sold = None
    best_ladder = None
    for ladder in candidates:
        sold = []
        total = 0.0
        for (capacity, fare, max_sell, noshow, threshold) in nights:
            s, val = best_sold_for_night(capacity, fare, max_sell, noshow, threshold, ladder)
            sold.append(s)
            total += val
        if total > best_total:
            best_total = total
            best_sold = sold
            best_ladder = ladder

    out = [" ".join(map(str, best_ladder)), " ".join(map(str, best_sold))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
