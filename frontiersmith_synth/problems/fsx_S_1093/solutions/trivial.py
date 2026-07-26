# TIER: trivial
"""Blind, re-searched binary-insertion certificate.

Maintains ONE global sorted array of ALL players registered so far (dead or alive --
it never notices that a cut player is now irrelevant) and inserts each new player via
ordinary binary search, filing EVERY comparison the search touches. This is a
perfectly correct certificate (any correct comparison-based sort's filed comparisons
transitively recover the whole order -- a classical fact), but it pays the full
"discover where I go" search cost for every single registrant, exactly reproducing the
checker's own internal baseline B. It never exploits that the whole script was known
in advance.
"""
import sys
import heapq


def main():
    data = sys.stdin.read().split()
    idx = [0]

    def nxt():
        v = data[idx[0]]
        idx[0] += 1
        return v

    n = int(nxt())
    q = int(nxt())
    ratings = [int(nxt()) for _ in range(n)]
    T = n + q
    events = [nxt() for _ in range(T)]

    out = []
    arr = []       # sorted ratings of all players registered so far (dead + alive)
    arr_pid = []   # parallel: player id at each position
    live_heap = []
    live = set()
    claimed = []
    enroll_count = 0

    for ev in events:
        if ev == 'E':
            enroll_count += 1
            pid = enroll_count
            r = ratings[pid - 1]
            lo, hi = 0, len(arr)
            while lo < hi:
                mid = (lo + hi) // 2
                out.append(('C', pid, arr_pid[mid]))
                if arr[mid] < r:
                    lo = mid + 1
                else:
                    hi = mid
            arr.insert(lo, r)
            arr_pid.insert(lo, pid)
            out.append(('M', pid))
            heapq.heappush(live_heap, (r, pid))
            live.add(pid)
        else:
            while live_heap and live_heap[0][1] not in live:
                heapq.heappop(live_heap)
            r, pid = heapq.heappop(live_heap)
            live.discard(pid)
            claimed.append(pid)

    lines = [' '.join(map(str, claimed))]
    for instr in out:
        if instr[0] == 'M':
            lines.append('M %d' % instr[1])
        else:
            lines.append('C %d %d' % (instr[1], instr[2]))
    sys.stdout.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
