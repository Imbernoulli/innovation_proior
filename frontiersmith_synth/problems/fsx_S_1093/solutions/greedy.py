# TIER: greedy
"""The obvious approach: "a priority queue IS a binary heap." Replay a real
array-based binary min-heap over the currently-registered-and-not-yet-cut players,
and log every internal comparison (sift-up on registration, sift-down on cut) as a
filed 'C'. This is a completely faithful, textbook live-scoreboard implementation --
it is exactly what a strong coder writes first, without pausing to notice that the
referee already has the whole script and doesn't need to re-derive the minimum from
scratch at every cut via O(log(live size)) fresh comparisons.
"""
import sys


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

    heap_pid = []  # array-based binary heap of player ids, ordered by rating

    def rating(pid):
        return ratings[pid - 1]

    out = []

    def sift_up(i):
        while i > 0:
            p = (i - 1) // 2
            out.append(('C', heap_pid[i], heap_pid[p]))
            if rating(heap_pid[i]) < rating(heap_pid[p]):
                heap_pid[i], heap_pid[p] = heap_pid[p], heap_pid[i]
                i = p
            else:
                break

    def sift_down(i):
        m = len(heap_pid)
        while True:
            l, r = 2 * i + 1, 2 * i + 2
            s = i
            if l < m:
                out.append(('C', heap_pid[l], heap_pid[s]))
                if rating(heap_pid[l]) < rating(heap_pid[s]):
                    s = l
            if r < m:
                out.append(('C', heap_pid[r], heap_pid[s]))
                if rating(heap_pid[r]) < rating(heap_pid[s]):
                    s = r
            if s == i:
                break
            heap_pid[i], heap_pid[s] = heap_pid[s], heap_pid[i]
            i = s

    claimed = []
    enroll_count = 0
    for ev in events:
        if ev == 'E':
            enroll_count += 1
            pid = enroll_count
            out.append(('M', pid))
            heap_pid.append(pid)
            sift_up(len(heap_pid) - 1)
        else:
            top = heap_pid[0]
            claimed.append(top)
            last = heap_pid.pop()
            if heap_pid:
                heap_pid[0] = last
                sift_down(0)

    lines = [' '.join(map(str, claimed))]
    for instr in out:
        if instr[0] == 'M':
            lines.append('M %d' % instr[1])
        else:
            lines.append('C %d %d' % (instr[1], instr[2]))
    sys.stdout.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
