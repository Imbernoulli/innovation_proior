# TIER: greedy
"""Myopic earliest-due-first dispatcher WITH chamber-batching, but no foresight
about the shared water budget: at every opportunity it simply serves whichever
direction currently holds the most urgent (smallest-due) waiting barge, batching
that direction up to capacity. This is the "obvious" first attempt -- it exploits
chamber-batching but keeps chasing due dates across both directions, so whenever
arrivals interleave tightly it ends up switching direction almost every cycle.
Direction switches cost far more water (wa >> ws) than staying put, so on the
family's trap instances this dispatcher burns through the reservoir in a handful
of cycles and stalls, stranding most of the fleet."""
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
    while unserved and guard < 20 * n + 50:
        guard += 1
        cur_time = prev_end
        available = [i for i in unserved if barges[i][0] <= cur_time]
        if not available:
            future = [barges[i][0] for i in unserved]
            nxt = min(future)
            if nxt < cur_time:
                nxt = cur_time
            cur_time = nxt
            available = [i for i in unserved if barges[i][0] <= cur_time]
            if not available:
                break

        by_dir = {0: [], 1: []}
        for i in available:
            by_dir[barges[i][1]].append(i)
        dir_order = sorted(
            [d for d in (0, 1) if by_dir[d]],
            key=lambda d: (min(barges[i][3] for i in by_dir[d]), d),
        )

        committed = False
        for d in dir_order:
            cand = sorted(by_dir[d], key=lambda i: (barges[i][3], i))
            batch = []
            tot_len = 0
            for i in cand:
                ln = barges[i][2]
                if len(batch) >= C:
                    break
                if tot_len + ln > L:
                    continue
                batch.append(i)
                tot_len += ln
            if not batch:
                continue
            s = cur_time
            if s + t > H:
                continue
            cost = ws if (prev_dir is None or d == prev_dir) else wa
            dt = s - prev_end
            W_after = W + rho * dt - cost
            if W_after < 0:
                continue
            W = W_after
            cycles.append((s, d, [i + 1 for i in batch]))
            for i in batch:
                unserved.discard(i)
            prev_end = s + t
            prev_dir = d
            committed = True
            break

        if not committed:
            future_after = [barges[i][0] for i in unserved if barges[i][0] > cur_time]
            if future_after:
                prev_end = min(future_after)
            else:
                break

    out = [str(len(cycles))]
    for (s, d, idxs) in cycles:
        out.append("%d %d %d %s" % (s, d, len(idxs), " ".join(map(str, idxs))))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
