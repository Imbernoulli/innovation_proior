# TIER: strong
# Decreasing-order packing under BOTH the thermal cap and the breaker-channel cap,
# keeping whichever of two classic policies energizes fewer transformers:
#   * first-fit-decreasing (FFD): steer the largest blocks first onto the lowest
#     transformer with room + a free channel;
#   * best-fit-decreasing (BFD): steer the largest blocks first onto the TIGHTEST
#     transformer that still fits, leaving the roomiest transformers open for
#     future heavy blocks.
# Sorting big-to-first lets small blocks top off partly loaded transformers, so
# waste drops well below the online rules -- but the loose dual-L1 bound keeps the
# normalized score below 1.0 on most instances.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
K = inst["channels"]
demands = inst["demands"]
N = len(demands)

order = sorted(range(N), key=lambda i: demands[i], reverse=True)


def first_fit_decreasing():
    rem = []
    cnt = []
    tof = [0] * N
    for i in order:
        d = demands[i]
        placed = -1
        for b in range(len(rem)):
            if rem[b] >= d and cnt[b] < K:
                placed = b
                break
        if placed < 0:
            rem.append(C - d)
            cnt.append(1)
            tof[i] = len(rem) - 1
        else:
            rem[placed] -= d
            cnt[placed] += 1
            tof[i] = placed
    return tof, len(rem)


def best_fit_decreasing():
    rem = []
    cnt = []
    tof = [0] * N
    for i in order:
        d = demands[i]
        best = -1
        best_rem = C + 1
        for b in range(len(rem)):
            if rem[b] >= d and cnt[b] < K and rem[b] < best_rem:
                best_rem = rem[b]
                best = b
        if best < 0:
            rem.append(C - d)
            cnt.append(1)
            tof[i] = len(rem) - 1
        else:
            rem[best] -= d
            cnt[best] += 1
            tof[i] = best
    return tof, len(rem)


ffd_assign, ffd_bins = first_fit_decreasing()
bfd_assign, bfd_bins = best_fit_decreasing()
assign = ffd_assign if ffd_bins <= bfd_bins else bfd_assign

print(json.dumps({"assign": assign}))
