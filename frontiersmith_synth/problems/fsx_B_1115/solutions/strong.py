# TIER: strong
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
    n = len(times)
    need_after = 0.0
    for i in range(n - 1, 0, -1):
        gap = times[i] - times[i - 1]
        factor = (1.0 - decay) ** gap
        need_before_i = amounts[i] + need_after
        need_after = need_before_i / factor
    return amounts[0] + need_after


def total_raw(k1, k2, k3, times, demands, S1, S2, S3, d1, d2, d3, want_events=False):
    g3 = group_indices(len(times), k3)
    lot3_t, lot3_draw, ev3 = [], [], []
    for (a, b) in g3:
        tt = times[a:b + 1]; dd = demands[a:b + 1]
        O3 = min_injection(tt, dd, d3)
        R3 = O3 + S3
        lot3_t.append(tt[0]); lot3_draw.append(R3)
        ev3.append((3, tt[0], R3))
    g2 = group_indices(len(lot3_t), k2)
    lot2_t, lot2_draw, ev2 = [], [], []
    for (a, b) in g2:
        tt = lot3_t[a:b + 1]; dd = lot3_draw[a:b + 1]
        O2 = min_injection(tt, dd, d2)
        R2 = O2 + S2
        lot2_t.append(tt[0]); lot2_draw.append(R2)
        ev2.append((2, tt[0], R2))
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

    # the genuine insight: decay compounds MULTIPLICATIVELY as material moves
    # through the chain, so the three stages' batching frequencies cannot be
    # tuned one at a time -- search the (k1,k2,k3) grid JOINTLY against the real
    # multi-buffer simulation. The winner is a matched RATIO of lot sizes between
    # adjacent stages, not any single stage's quantity chosen in isolation.
    best = None
    for k3 in range(1, T + 1):
        for k2 in range(1, T + 1):
            for k1 in range(1, T + 1):
                v = total_raw(k1, k2, k3, times, demands, S1, S2, S3, d1, d2, d3)
                if best is None or v < best[0]:
                    best = (v, k1, k2, k3)

    _, k1, k2, k3 = best
    _, events = total_raw(k1, k2, k3, times, demands, S1, S2, S3, d1, d2, d3, want_events=True)

    lines = [str(len(events))]
    for (s, t, r) in events:
        lines.append("%d %d %.6f" % (s, t, r))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
