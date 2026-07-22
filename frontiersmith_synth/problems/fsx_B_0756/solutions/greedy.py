# TIER: greedy
"""The obvious recipe: one global average no-show rate sets the same aggressive sale
cap on every night, and the ladder is built from pooled percentiles of ALL thresholds.
Blind to the correlation between high-fare nights, low no-show, and high thresholds."""
import sys


def main():
    toks = sys.stdin.read().split()
    pos = 0
    n = int(toks[pos]); pos += 1
    ladder_lo, ladder_hi, penalty = int(toks[pos]), int(toks[pos + 1]), int(toks[pos + 2])
    pos += 3
    nights = []
    all_noshow = []
    all_thresholds = []
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
            all_noshow.append(ns)
            all_thresholds.append(th)
        nights.append((capacity, fare, max_sell, noshow, threshold))

    # the average no-show rate (pooled over every printed passenger everywhere) says the
    # season is safe to overbook -- so sell every night out to its printed headroom cap
    q_avg = sum(all_noshow) / max(1, len(all_noshow))
    q_avg = min(0.9, max(0.01, q_avg))  # computed, but the recipe below ignores per-night gaps

    sold = [max_sell for (capacity, fare, max_sell, noshow, threshold) in nights]

    # textbook ladder centered on the pooled MEAN and standard deviation -- treats the
    # threshold population as one roughly-unimodal group instead of reading off the
    # actual tail quantile, so it badly undershoots the high-threshold minority
    m = len(all_thresholds)
    mean_th = sum(all_thresholds) / max(1, m)
    var_th = sum((x - mean_th) ** 2 for x in all_thresholds) / max(1, m)
    std_th = var_th ** 0.5

    raw = [mean_th - 0.6 * std_th, mean_th - 0.1 * std_th, mean_th + 0.4 * std_th,
           mean_th + 0.9 * std_th, mean_th + 1.3 * std_th]
    ladder = sorted(max(ladder_lo, min(ladder_hi, int(round(v)))) for v in raw)
    for i in range(1, 5):
        if ladder[i] <= ladder[i - 1]:
            ladder[i] = ladder[i - 1] + 1
    if ladder[4] > ladder_hi:
        ladder[4] = ladder_hi
        for i in range(3, -1, -1):
            if ladder[i] >= ladder[i + 1]:
                ladder[i] = ladder[i + 1] - 1

    out = [" ".join(map(str, ladder)), " ".join(map(str, sold))]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
