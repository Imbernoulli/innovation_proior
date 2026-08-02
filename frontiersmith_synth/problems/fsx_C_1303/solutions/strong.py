# TIER: strong
# Insight: the whole demand-mix schedule is given up front, so instead of
# myopically re-pointing the boost pool at whichever station is slowest THIS
# epoch (which pays a changeover every time the bottleneck migrates and can
# even move boost to a station that barely benefits from it), we:
#   1) DP-segment the known horizon into a handful of contiguous "regimes"
#      that each keep ONE static boost allocation (regimes only switch when
#      the future bottleneck is worth paying a changeover for) -- this
#      anticipates where the bottleneck is headed rather than chasing where
#      it currently is;
#   2) size the one-time buffer budget by the imbalance the CHOSEN schedule
#      will actually produce at each gap (concentrating buffer where the
#      migration/changeover stress is worst), not by an even split;
#   3) evaluate a small menu of (segmentation, buffer-plan) combinations with
#      a faithful re-implementation of the real tick-by-tick flow simulation
#      and keep the best one.
import sys, json


def eff_cycle(base, k, units):
    return base / (1.0 + k * units)


def simulate(K, T, L, base_cycle, k_eff, buffers, alloc, d0, d1, m0, m1, initial_alloc):
    levels = [0.0] * (K - 1)
    shipped = 0.0
    money = 0.0
    prev = list(initial_alloc)
    for t in range(T):
        row = alloc[t]
        downtime = [0] * K
        for i in range(K):
            d = abs(row[i] - prev[i])
            if d > 0:
                dt = d0[i] + d1[i] * d
                downtime[i] = min(L[t] - 1, int(round(dt)))
                money += m0[i] + m1[i] * d
        prev = row
        cyc = [eff_cycle(base_cycle[t][i], k_eff[i], row[i]) for i in range(K)]
        rate = [(1.0 / c) if c > 1e-9 else 0.0 for c in cyc]
        for tick in range(L[t]):
            for i in range(K - 1, -1, -1):
                r = 0.0 if tick < downtime[i] else rate[i]
                inp = levels[i - 1] if i > 0 else float("inf")
                space = (buffers[i] - levels[i]) if i < K - 1 else float("inf")
                produce = min(r, inp, space)
                if produce < 0.0:
                    produce = 0.0
                if i > 0:
                    levels[i - 1] -= produce
                if i < K - 1:
                    levels[i] += produce
                else:
                    shipped += produce
    return shipped, money


def compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in compositions(total - first, parts - 1):
            yield (first,) + rest


def best_static_alloc(K, base_cycle_rows, L_rows, k_eff, P, all_comps):
    """Best single allocation (ignoring changeover/buffers) for a contiguous
    block of epochs, judged by ideal throughput on that block."""
    best_val, best_row = -1.0, tuple([P // K] * K)
    for row in all_comps:
        val = 0.0
        for bc, ln in zip(base_cycle_rows, L_rows):
            cyc = [eff_cycle(bc[i], k_eff[i], row[i]) for i in range(K)]
            val += ln * min(1.0 / c for c in cyc)
        if val > best_val:
            best_val, best_row = val, row
    return list(best_row), best_val


def uniform_buffers(K, budget):
    n = K - 1
    base = budget // n
    rem = budget % n
    return [base + (1 if j < rem else 0) for j in range(n)]


def stress_buffers(K, budget, T, alloc, base_cycle, k_eff, initial_alloc):
    """Weight each gap by how imbalanced its two neighbor stations' effective
    rates are across the chosen schedule (including the epoch right after any
    changeover, where the imbalance is worst)."""
    n = K - 1
    weight = [1.0] * n
    prev = list(initial_alloc)
    for t in range(T):
        row = alloc[t]
        changed = [1.0 if row[i] != prev[i] else 0.0 for i in range(K)]
        rate = []
        for i in range(K):
            c = eff_cycle(base_cycle[t][i], k_eff[i], row[i])
            rate.append(1.0 / c if c > 1e-9 else 0.0)
        for j in range(n):
            mismatch = abs(rate[j] - rate[j + 1])
            bump = 1.0 + 2.0 * max(changed[j], changed[j + 1])
            weight[j] += mismatch * bump
        prev = row
    total_w = sum(weight)
    raw = [budget * w / total_w for w in weight]
    bufs = [max(1, int(x)) for x in raw]
    # fix rounding so sum <= budget, floor 1 each, extra goes to the highest-weight gap
    while sum(bufs) > budget:
        j = max(range(n), key=lambda j: bufs[j])
        if bufs[j] > 1:
            bufs[j] -= 1
        else:
            break
    while sum(bufs) < budget:
        j = max(range(n), key=lambda j: weight[j])
        bufs[j] += 1
    return bufs


def main():
    inst = json.load(sys.stdin)
    K, T, P = inst["K"], inst["T"], inst["P"]
    L = inst["L"]
    base_cycle = inst["base_cycle"]
    k_eff = inst["k_eff"]
    budget = inst["buffer_budget"]
    d0, d1, m0, m1 = inst["changeover_downtime_fixed"], inst["changeover_downtime_per_unit"], \
        inst["changeover_money_fixed"], inst["changeover_money_per_unit"]
    initial_alloc = inst["initial_alloc"]
    money_weight = inst.get("money_weight", 1.0)

    all_comps = list(compositions(P, K))

    # precompute best static allocation + ideal value for every contiguous subrange
    best_row = [[None] * (T + 1) for _ in range(T + 1)]
    best_val = [[0.0] * (T + 1) for _ in range(T + 1)]
    for a in range(T):
        for b in range(a + 1, T + 1):
            row, val = best_static_alloc(K, base_cycle[a:b], L[a:b], k_eff, P, all_comps)
            best_row[a][b] = row
            best_val[a][b] = val

    # changeover penalty estimate for switching TO `row` FROM `prev_row` (used inside the DP
    # so it prefers segmentations that don't pay for moves that aren't worth it)
    def switch_cost(prev_row, row):
        c = 0.0
        for i in range(K):
            d = abs(row[i] - prev_row[i])
            if d > 0:
                c += m0[i] + m1[i] * d
                c += (d0[i] + d1[i] * d) * 0.6  # rough throughput-tick penalty, in "money units"
        return c

    max_seg = min(T, 6)
    candidates = []  # list of (alloc_schedule)

    for s in range(1, max_seg + 1):
        # DP over number of segments s: dp[t] = (value, split_points, chosen rows)
        NEG = float("-inf")
        dp = [[NEG] * (s + 1) for _ in range(T + 1)]
        choice = [[None] * (s + 1) for _ in range(T + 1)]
        dp[0][0] = 0.0
        for t in range(1, T + 1):
            for seg in range(1, s + 1):
                for a in range(seg - 1, t):
                    if dp[a][seg - 1] == NEG:
                        continue
                    row = best_row[a][t]
                    prev_row = choice[a][seg - 1][2] if a > 0 and choice[a][seg - 1] else initial_alloc
                    val = dp[a][seg - 1] + best_val[a][t] - switch_cost(prev_row, row)
                    if val > dp[t][seg]:
                        dp[t][seg] = val
                        choice[t][seg] = (a, seg - 1, row)
        if dp[T][s] == NEG:
            continue
        # reconstruct
        segs = []
        t, seg = T, s
        while seg > 0:
            a, prev_seg, row = choice[t][seg]
            segs.append((a, t, row))
            t, seg = a, prev_seg
        segs.reverse()
        alloc = [None] * T
        for a, b, row in segs:
            for tt in range(a, b):
                alloc[tt] = list(row)
        candidates.append(alloc)

    # also throw in pure greedy (chase current bottleneck every epoch) as a candidate,
    # in case some instance genuinely wants it (e.g. a stable single bottleneck)
    greedy_alloc = []
    for t in range(T):
        row = [0] * K
        b = max(range(K), key=lambda i: base_cycle[t][i])
        row[b] = P
        greedy_alloc.append(row)
    candidates.append(greedy_alloc)

    best_overall = None
    best_score = float("-inf")
    for alloc in candidates:
        for bufs in (uniform_buffers(K, budget),
                     stress_buffers(K, budget, T, alloc, base_cycle, k_eff, initial_alloc)):
            shipped, money = simulate(K, T, L, base_cycle, k_eff, bufs, alloc, d0, d1, m0, m1, initial_alloc)
            obj = shipped - money_weight * money
            if obj > best_score:
                best_score = obj
                best_overall = (alloc, bufs)

    alloc, bufs = best_overall
    print(json.dumps({"alloc": alloc, "buffers": bufs}))


if __name__ == "__main__":
    main()
