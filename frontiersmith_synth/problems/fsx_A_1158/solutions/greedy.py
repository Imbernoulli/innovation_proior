# TIER: greedy
# The "obvious" first attempt: sort ships by DESTINATION (nearest first) and pack
# consecutive groups of up to c ships into escorted convoys, dispatched round-robin
# across breakers as soon as each breaker is free. Never leases the clear wake, and
# never groups by speed -- so convoys routinely mix fast and slow ships, throttling
# the whole batch to its slowest member.
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

    order = sorted(ships, key=lambda x: (x[1], x[3]))
    batches = [order[i:i + c] for i in range(0, M, c)]

    avail = [0] * B
    trips = []
    for bi, batch in enumerate(batches):
        b = bi % B
        t0 = avail[b]
        maxD = max(x[1] for x in batch)
        cur = t0
        for cell in range(1, maxD + 1):
            active = [x[0] for x in batch if x[1] >= cell]
            step = max(1, max(active))
            cur += step
        finish = cur
        avail[b] = finish + maxD
        ids = [x[3] for x in batch]
        trips.append((b, t0, ids))

    out = [str(len(trips))]
    for (b, t0, ids) in trips:
        out.append("%d %d %d %s" % (b, t0, len(ids), " ".join(map(str, ids))))
    out.append("0")
    sys.stdout.write("\n".join(out) + "\n")


main()
