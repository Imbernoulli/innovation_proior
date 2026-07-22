# TIER: trivial
"""FCFS-with-batching: repeatedly serve the direction of the earliest-arrived
still-waiting barge, batching its same-direction, earliest-arrived companions up
to chamber capacity. No idea about the water budget or due dates at all -- this
is exactly the checker's internal baseline construction, and typically stalls
partway through the fleet once the reservoir (or the horizon) runs out."""
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); C = int(next(it)); L = int(next(it)); t = int(next(it))
    H = int(next(it)); W0 = int(next(it)); rho = int(next(it))
    ws = int(next(it)); wa = int(next(it))
    barges = []
    for i in range(n):
        a = int(next(it)); d = int(next(it)); ln = int(next(it))
        due = int(next(it)); wt = int(next(it))
        barges.append((a, d, ln, due, wt))

    unserved = set(range(n))
    prev_end = 0
    prev_dir = None
    W = W0
    cycles = []
    guard = 0
    while unserved and guard < 4 * n + 20:
        guard += 1
        cur_time = prev_end
        avail = [i for i in unserved if barges[i][0] <= cur_time]
        if not avail:
            nxt = min(barges[i][0] for i in unserved)
            cur_time = max(cur_time, nxt)
            avail = [i for i in unserved if barges[i][0] <= cur_time]
            if not avail:
                break
        avail.sort(key=lambda i: (barges[i][0], i))
        d = barges[avail[0]][1]
        cand = sorted((i for i in avail if barges[i][1] == d),
                      key=lambda i: (barges[i][0], i))
        batch = []
        tot_len = 0
        for i in cand:
            if len(batch) >= C:
                break
            ln = barges[i][2]
            if tot_len + ln <= L:
                batch.append(i)
                tot_len += ln
        if not batch:
            break
        s = cur_time
        if s + t > H:
            break
        cost = ws if (prev_dir is None or d == prev_dir) else wa
        dt = s - prev_end
        W_after = W + rho * dt - cost
        if W_after < 0:
            break
        W = W_after
        for i in batch:
            unserved.discard(i)
        cycles.append((s, d, [i + 1 for i in batch]))
        prev_end = s + t
        prev_dir = d

    out = [str(len(cycles))]
    for (s, d, idxs) in cycles:
        out.append("%d %d %d %s" % (s, d, len(idxs), " ".join(map(str, idxs))))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
