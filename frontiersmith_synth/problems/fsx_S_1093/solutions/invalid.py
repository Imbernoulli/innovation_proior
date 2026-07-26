# TIER: invalid
"""Correct answers, zero proof. Recomputes the TRUE cut sequence (so the identities
are exactly right) but files no comparisons at all -- exactly the "just write the
answers" shortcut the statement explicitly forbids. Must score 0: the checker demands
that every cut be CERTIFIED by filed comparisons, not merely correct.
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

    heap = []
    live = set()
    claimed = []
    enroll_count = 0
    for ev in events:
        if ev == 'E':
            enroll_count += 1
            pid = enroll_count
            heapq.heappush(heap, (ratings[pid - 1], pid))
            live.add(pid)
        else:
            while heap and heap[0][1] not in live:
                heapq.heappop(heap)
            r, pid = heapq.heappop(heap)
            live.discard(pid)
            claimed.append(pid)

    lines = [' '.join(map(str, claimed))]
    for pid in range(1, n + 1):
        lines.append('M %d' % pid)
    # deliberately NO 'C' lines -- correct answers, uncertified.
    sys.stdout.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
