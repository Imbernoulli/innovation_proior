#!/usr/bin/env python3
"""gen.py <testId> -- prints ONE instance of fsx_B_1283 to stdout.

Format:
  T
  n_starts
  starts[0..n_starts-1]
  MAX_PER_SLOT
  cost_base_J cost_base_S
  ot_num_J ot_den_J ot_num_S ot_den_S
  cost_ot_J cost_ot_S
  cost_agency_J cost_agency_S
  K
  (K blocks of T lines "L H")

All 10 cases are fully determined by testId (seeded via testId only). The
ward runs T=24 hourly slots, staffed via n_starts=3 non-overlapping 8-hour
shift blocks (starts = 0, 8, 16 -- night/day/evening). Block 2 (hours 16..23)
is the "evening" block where the predictable surge lives.

Cases 3,4,5,7,9,10 are TRAP cases: a majority-but-not-all of the K held-out
days carry a sharp evening acuity-skewed surge (low baseline the rest of the
time). Averaging demand across all K days (the obvious first move) dilutes
the surge's true height by the non-surge days mixed into the mean, so a
roster sized off the mean systematically under-provisions senior capacity
for the surge days specifically -- forcing expensive senior agency exactly
when it is least necessary (the surge is the SAME recurring shape every
time it happens). Cases 9-10 additionally rotate the surge's exact window
across different held-out days (17-21 vs 19-23) so a roster fixed once must
cover the union via flexible capacity, not one narrow guess.
"""
import sys, random

T = 24
STARTS = [0, 8, 16]
BLOCK_LEN = 8

COST_BASE_J, COST_BASE_S = 56, 70
OT_NUM_J, OT_DEN_J = 1, 2
OT_NUM_S, OT_DEN_S = 1, 3
COST_OT_J, COST_OT_S = 11, 18
COST_AGENCY_J, COST_AGENCY_S = 40, 80

LOW_BASE, HIGH_BASE = 5, 2

# testId -> (K, MAX_PER_SLOT, peak_day_count, peak_windows, low_mult, high_mult, jitter)
# peak_windows: list of (start_hour, length) candidates rotated across the peaked days
# Trap cases (>=50% of days peaked, so a distribution-aware estimator reliably
# sees the surge) are 3, 4, 5, 7, 9, 10. Control cases (0, or a rare <=25%
# minority day that no reasonable estimator would size against) are 1, 2, 6, 8.
TABLE = {
    1:  (3,  12, 0, [], 1.0, 1.0, 1),
    2:  (5,  14, 1, [(18, 4)], 1.8, 3.0, 1),
    3:  (6,  17, 3, [(18, 5)], 2.5, 6.0, 2),
    4:  (7,  19, 4, [(18, 5)], 2.6, 6.5, 2),
    5:  (8,  21, 4, [(17, 4), (20, 4)], 2.7, 7.0, 2),
    6:  (6,  18, 1, [(19, 3)], 2.0, 4.0, 2),
    7:  (8,  23, 4, [(19, 4)], 3.2, 8.0, 2),
    8:  (9,  24, 2, [(18, 4)], 2.2, 4.5, 2),
    9:  (10, 27, 5, [(17, 4), (20, 4)], 2.9, 7.5, 3),
    10: (12, 30, 6, [(16, 3), (19, 3), (21, 3)], 3.3, 8.5, 3),
}


def gen_day(rng, is_peak, window, low_mult, high_mult, jitter):
    L = [0] * T
    H = [0] * T
    for t in range(T):
        l = LOW_BASE + rng.randint(-1, 1)
        h = HIGH_BASE + rng.randint(-1, 1)
        L[t] = max(0, l)
        H[t] = max(0, h)
    if is_peak:
        ws, wl = window
        for off in range(wl):
            t = (ws + off) % T
            l = int(round(LOW_BASE * low_mult)) + rng.randint(-jitter, jitter)
            h = int(round(HIGH_BASE * high_mult)) + rng.randint(-jitter, jitter)
            L[t] = max(0, l)
            H[t] = max(0, h)
    return L, H


def main():
    t_id = int(sys.argv[1])
    K, max_per_slot, peak_count, windows, low_mult, high_mult, jitter = TABLE[t_id]
    rng = random.Random(20000 + 97 * t_id)

    print(T)
    print(len(STARTS))
    print(*STARTS)
    print(max_per_slot)
    print(COST_BASE_J, COST_BASE_S)
    print(OT_NUM_J, OT_DEN_J, OT_NUM_S, OT_DEN_S)
    print(COST_OT_J, COST_OT_S)
    print(COST_AGENCY_J, COST_AGENCY_S)
    print(K)

    # decide which of the K days are peaked (spread deterministically, not just the
    # first `peak_count` days, so ordering carries no exploitable signal) and which
    # window each uses (rotating through `windows` when more than one is given).
    day_order = list(range(K))
    rng.shuffle(day_order)
    peak_days = set(day_order[:peak_count])

    for d in range(K):
        is_peak = d in peak_days
        window = windows[d % len(windows)] if (is_peak and windows) else (0, 0)
        L, H = gen_day(rng, is_peak, window, low_mult, high_mult, jitter)
        for tt in range(T):
            print(L[tt], H[tt])


if __name__ == "__main__":
    main()
