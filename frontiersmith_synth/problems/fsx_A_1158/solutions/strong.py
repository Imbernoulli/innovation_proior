# TIER: strong
# The insight: the refreeze timer turns a cleared corridor into a LEASED resource that
# many ships can ride concurrently, so capacity c is only a bottleneck for the ships that
# actually need direct escort. Bucket ships by exact speed (homogeneous convoys never
# throttle each other). Per bucket: if it fits within capacity, escort it whole. Otherwise
# escort only up to c members -- always including whichever ship reaches farthest (so the
# corridor is cleared out to the whole bucket's need) plus the highest weight*destination
# members -- and let every other same-speed ship in the bucket LEASE that pilot's clear
# wake for free, departing at the pilot's own tick (always inside the refreeze window since
# same speed keeps the gap constant at zero). Buckets are dispatched fastest-first,
# round-robin across breakers.
import sys


def main():
    data = sys.stdin.read().split()
    p = 0
    L = int(data[p]); p += 1
    M = int(data[p]); p += 1
    B = int(data[p]); p += 1
    c = int(data[p]); p += 1
    r = int(data[p]); p += 1
    ships = []
    for j in range(M):
        s = int(data[p]); p += 1
        d = int(data[p]); p += 1
        w = int(data[p]); p += 1
        ships.append((s, d, w, j + 1))

    buckets = {}
    for sh in ships:
        buckets.setdefault(sh[0], []).append(sh)

    plan = []  # (escort_list, piggy_list)
    for sp in sorted(buckets.keys()):
        lst = buckets[sp]
        if len(lst) <= c:
            plan.append((lst, []))
        else:
            far = max(lst, key=lambda x: x[1])
            rest = [x for x in lst if x is not far]
            rest_sorted = sorted(rest, key=lambda x: -(x[2] * x[1]))
            escort = [far] + rest_sorted[:c - 1]
            escort_ids = set(x[3] for x in escort)
            piggy = [x for x in lst if x[3] not in escort_ids]
            plan.append((escort, piggy))

    avail = [0] * B
    trips = []
    piggy_out = []
    for bi, (escort, piggy) in enumerate(plan):
        b = bi % B
        t0 = avail[b]
        maxD = max(x[1] for x in escort)
        cur = t0
        for cell in range(1, maxD + 1):
            active = [x[0] for x in escort if x[1] >= cell]
            step = max(1, max(active))
            cur += step
        finish = cur
        avail[b] = finish + maxD
        ids = [x[3] for x in escort]
        trips.append((b, t0, ids))
        for x in piggy:
            piggy_out.append((x[3], t0))  # same tick as pilot: Delta=0, always inside [0, r)

    out = [str(len(trips))]
    for (b, t0, ids) in trips:
        out.append("%d %d %d %s" % (b, t0, len(ids), " ".join(map(str, ids))))
    out.append(str(len(piggy_out)))
    for (sid, t0) in piggy_out:
        out.append("%d %d" % (sid, t0))
    sys.stdout.write("\n".join(out) + "\n")


main()
