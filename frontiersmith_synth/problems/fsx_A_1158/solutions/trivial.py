# TIER: trivial
# Escort every ship ALONE (its own convoy of size 1), round-robin across the B breakers
# in ship-index order, each departing the instant its breaker is free. No batching, no
# leasing. Reproduces the checker's own internal baseline construction exactly.
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
    for _ in range(M):
        s = int(data[p]); p += 1
        d = int(data[p]); p += 1
        w = int(data[p]); p += 1
        ships.append((s, d, w))

    avail = [0] * B
    trips = []
    for j in range(1, M + 1):
        s, d, w = ships[j - 1]
        b = (j - 1) % B
        t0 = avail[b]
        arrival = t0 + d * s
        avail[b] = arrival + d
        trips.append((b, t0, j))

    out = [str(len(trips))]
    for (b, t0, sid) in trips:
        out.append("%d %d 1 %d" % (b, t0, sid))
    out.append("0")
    sys.stdout.write("\n".join(out) + "\n")


main()
