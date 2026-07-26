# TIER: strong
"""The insight: don't replay a container -- file the minimal certificate.

The referee already has the whole script, so "where does the new registrant sit
relative to everyone seen so far" is a FREE internal lookup (binary search over an
internal sorted array costs nothing -- only what gets FILED costs anything). So for
each new registrant we look up its two immediate rating-neighbors among everyone
registered so far (dead or alive) and file only those 1-2 comparisons, linking it into
a single global chain.

Why this is a complete, correct certificate: any correct comparison-based sorting
algorithm's filed comparisons must, by transitivity, recover the FULL total order (the
standard adversary argument behind comparison-sort lower bounds) -- and by induction,
linking every new element only to its current immediate predecessor/successor already
achieves that, because each earlier pair (x, y) with x<y remains connected through the
chain of intermediate elements that were, in turn, linked to THEIR neighbors when they
arrived. So every cut is certified "for free" the moment it happens: the current chain
head already transitively dominates everyone else live, no extra comparisons needed at
cut time at all. This costs ~1-2 comparisons per registration total, versus a live
heap's O(log(live size)) *per operation* (registration AND cut) -- the search cost a
container pays to *discover* structure it doesn't need to discover, because we already
know it.
"""
import sys
import bisect


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
    all_ratings = []  # sorted ratings of everyone registered so far (dead + alive)
    all_pid = []      # parallel player ids

    # separate small bookkeeping structure to answer "who is currently live-minimum"
    # for free (internal only, never filed) -- kept as a sorted array of the LIVE set.
    live_ratings = []
    live_pid = []

    claimed = []
    enroll_count = 0
    for ev in events:
        if ev == 'E':
            enroll_count += 1
            pid = enroll_count
            r = ratings[pid - 1]

            pos = bisect.bisect_left(all_ratings, r)
            if pos > 0:
                out.append(('C', all_pid[pos - 1], pid))     # predecessor below pid
            if pos < len(all_ratings):
                out.append(('C', pid, all_pid[pos]))         # pid below successor
            all_ratings.insert(pos, r)
            all_pid.insert(pos, pid)
            out.append(('M', pid))

            lpos = bisect.bisect_left(live_ratings, r)
            live_ratings.insert(lpos, r)
            live_pid.insert(lpos, pid)
        else:
            live_ratings.pop(0)
            pid = live_pid.pop(0)
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
