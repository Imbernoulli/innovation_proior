# TIER: greedy
import sys


def group_indices(n, k):
    groups = []
    i = 0
    while i < n:
        j = min(i + k, n) - 1
        groups.append((i, j))
        i = j + 1
    return groups


def min_injection(times, amounts, decay):
    """Minimal quantity to inject at times[0] (same-hour, zero decay before the
    first withdrawal) so a pool fed only by that single injection survives the
    chain of withdrawals amounts[i] at times[i] (i=0..n-1), given the pool decays
    by a factor (1-decay) per elapsed hour between withdrawals."""
    n = len(times)
    need_after = 0.0
    for i in range(n - 1, 0, -1):
        gap = times[i] - times[i - 1]
        factor = (1.0 - decay) ** gap
        need_before_i = amounts[i] + need_after
        need_after = need_before_i / factor
    return amounts[0] + need_after


def total_raw(k1, k2, k3, times, demands, S1, S2, S3, d1, d2, d3, want_events=False):
    # level 3: one stage-3 lot per window of k3 consecutive pulses
    g3 = group_indices(len(times), k3)
    lot3_t, lot3_draw, ev3 = [], [], []
    for (a, b) in g3:
        tt = times[a:b + 1]; dd = demands[a:b + 1]
        O3 = min_injection(tt, dd, d3)
        R3 = O3 + S3
        lot3_t.append(tt[0]); lot3_draw.append(R3)
        ev3.append((3, tt[0], R3))
    # level 2: one stage-2 lot per group of k2 stage-3 lots
    g2 = group_indices(len(lot3_t), k2)
    lot2_t, lot2_draw, ev2 = [], [], []
    for (a, b) in g2:
        tt = lot3_t[a:b + 1]; dd = lot3_draw[a:b + 1]
        O2 = min_injection(tt, dd, d2)
        R2 = O2 + S2
        lot2_t.append(tt[0]); lot2_draw.append(R2)
        ev2.append((2, tt[0], R2))
    # level 1: one stage-1 lot per group of k1 stage-2 lots
    g1 = group_indices(len(lot2_t), k1)
    ev1 = []
    total = 0.0
    for (a, b) in g1:
        tt = lot2_t[a:b + 1]; dd = lot2_draw[a:b + 1]
        O1 = min_injection(tt, dd, d1)
        R1 = O1 + S1
        ev1.append((1, tt[0], R1))
        total += R1
    if want_events:
        return total, ev1 + ev2 + ev3
    return total


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it))
    times = [int(next(it)) for _ in range(T)]
    demands = [int(next(it)) for _ in range(T)]
    S1 = int(next(it)); S2 = int(next(it)); S3 = int(next(it))
    d1 = float(next(it)); d2 = float(next(it)); d3 = float(next(it))

    # per-stage-isolated EOQ: pick each stage's own best batching frequency
    # assuming the OTHER two stages stay just-in-time (the obvious textbook move
    # -- optimize the stage you're looking at, treat the rest as pass-through),
    # then combine the three independently-chosen frequencies into one schedule.
    def best_k1():
        best_k, best_v = 1, total_raw(1, 1, 1, times, demands, S1, S2, S3, d1, d2, d3)
        for k in range(2, T + 1):
            v = total_raw(k, 1, 1, times, demands, S1, S2, S3, d1, d2, d3)
            if v < best_v:
                best_v, best_k = v, k
        return best_k

    def best_k2():
        best_k, best_v = 1, total_raw(1, 1, 1, times, demands, S1, S2, S3, d1, d2, d3)
        for k in range(2, T + 1):
            v = total_raw(1, k, 1, times, demands, S1, S2, S3, d1, d2, d3)
            if v < best_v:
                best_v, best_k = v, k
        return best_k

    def best_k3():
        best_k, best_v = 1, total_raw(1, 1, 1, times, demands, S1, S2, S3, d1, d2, d3)
        for k in range(2, T + 1):
            v = total_raw(1, 1, k, times, demands, S1, S2, S3, d1, d2, d3)
            if v < best_v:
                best_v, best_k = v, k
        return best_k

    k1 = best_k1(); k2 = best_k2(); k3 = best_k3()
    _, events = total_raw(k1, k2, k3, times, demands, S1, S2, S3, d1, d2, d3, want_events=True)

    lines = [str(len(events))]
    for (s, t, r) in events:
        lines.append("%d %d %.6f" % (s, t, r))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
