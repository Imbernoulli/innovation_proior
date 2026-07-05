# TIER: strong
# Decreasing-order packing with two classic policies, keep whichever dispatches
# fewer gondolas:
#   * first-fit-decreasing (FFD): seat the largest parties first into the lowest
#     gondola that fits;
#   * best-fit-decreasing (BFD): seat the largest parties first into the TIGHTEST
#     gondola that still fits (leaves the most usable large gaps open).
# Sorting big-to-first lets small parties top off partially-filled gondolas, so
# waste drops well below the online rules -- but the loose L1 bound keeps the
# normalized score below 1.0 on most instances.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
parties = inst["parties"]
N = len(parties)

order = sorted(range(N), key=lambda i: parties[i], reverse=True)


def first_fit_decreasing():
    rem = []
    gof = [0] * N
    for i in order:
        s = parties[i]
        placed = -1
        for b in range(len(rem)):
            if rem[b] >= s:
                placed = b
                break
        if placed < 0:
            rem.append(C - s)
            gof[i] = len(rem) - 1
        else:
            rem[placed] -= s
            gof[i] = placed
    return gof, len(rem)


def best_fit_decreasing():
    rem = []
    gof = [0] * N
    for i in order:
        s = parties[i]
        best = -1
        best_rem = C + 1
        for b in range(len(rem)):
            if rem[b] >= s and rem[b] < best_rem:
                best_rem = rem[b]
                best = b
        if best < 0:
            rem.append(C - s)
            gof[i] = len(rem) - 1
        else:
            rem[best] -= s
            gof[i] = best
    return gof, len(rem)


ffd_assign, ffd_bins = first_fit_decreasing()
bfd_assign, bfd_bins = best_fit_decreasing()
assign = ffd_assign if ffd_bins <= bfd_bins else bfd_assign

print(json.dumps({"assign": assign}))
