#!/usr/bin/env python3
"""gen.py <testId> -- prints one sleeper-train overbooking instance to stdout.
Deterministic: all randomness seeded from testId only.

Format:
  N
  LADDER_LO LADDER_HI PENALTY
  for each of N nights:
    capacity fare max_sell
    2*max_sell ints: noshow_1 threshold_1 noshow_2 threshold_2 ... noshow_max_sell threshold_max_sell
       (noshow_j in {0,1}; threshold_j is the compensation at which passenger j -- the j-th ticket
        sold that night, in sale order -- would voluntarily give up their berth if bumped)

Every 5th night (idx % 5 == 2, 0-indexed) is a PEAK night: high fare, LOW no-show rate, HIGH
volunteer thresholds -- the correlated trap the statement warns about.
"""
import random
import sys

N_LIST = [10, 14, 18, 24, 30, 40, 50, 62, 76, 90]

LADDER_LO = 10
LADDER_HI = 750
PENALTY = 2200


def build_night(rng, is_peak):
    capacity = rng.randint(30, 60)
    if is_peak:
        # peak nights are printed with generous overbooking headroom (they sell out fast,
        # so the season planner authorizes more room) -- exactly what makes a mean-based
        # cap dangerous there
        headroom = max(6, round(capacity * 0.42))
        max_sell = capacity + headroom
        fare = rng.randint(380, 600)
        noshow_rate = rng.uniform(0.02, 0.06)
        thr_lo, thr_hi = int(fare * 0.75), int(fare * 1.15)
    else:
        headroom = max(4, round(capacity * 0.16))
        max_sell = capacity + headroom
        fare = rng.randint(80, 300)
        noshow_rate = rng.uniform(0.15, 0.30)
        thr_lo, thr_hi = max(1, int(fare * 0.20)), max(2, int(fare * 0.60))
    noshow = []
    threshold = []
    for _ in range(max_sell):
        noshow.append(1 if rng.random() < noshow_rate else 0)
        threshold.append(rng.randint(thr_lo, thr_hi))
    return capacity, fare, max_sell, noshow, threshold


def main():
    test_id = int(sys.argv[1])
    n = N_LIST[test_id - 1]
    rng = random.Random(900001 + 733 * test_id)

    out = [str(n), f"{LADDER_LO} {LADDER_HI} {PENALTY}"]
    for idx in range(n):
        is_peak = (idx % 5 == 2)
        capacity, fare, max_sell, noshow, threshold = build_night(rng, is_peak)
        out.append(f"{capacity} {fare} {max_sell}")
        toks = []
        for j in range(max_sell):
            toks.append(str(noshow[j]))
            toks.append(str(threshold[j]))
        out.append(" ".join(toks))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
