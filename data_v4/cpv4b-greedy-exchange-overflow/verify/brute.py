import sys
from functools import lru_cache

# Independent, obviously-correct brute force for the job-sequencing problem.
#
# Problem: n performances; performance i earns profit p[i] and must be staged in
# exactly one time slot in {1, 2, ..., d[i]}. Each slot hosts at most one
# performance, each performance takes one slot. Maximize total profit (you may
# leave performances unscheduled).
#
# Exact method (no greedy assumed): recurse over performances. For each one we
# either skip it or place it in any free slot in 1..d[i]. The set of busy slots
# is a bitmask over slots 1..maxd. Memoized on (index, busy_mask). For the small
# random tests (n, deadlines small) this enumerates the optimum exactly.

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        print(0)
        return
    it = iter(data)
    n = int(next(it))
    jobs = []
    for _ in range(n):
        p = int(next(it)); d = int(next(it))
        jobs.append((p, d))

    if n == 0:
        print(0)
        return

    sys.setrecursionlimit(1000000)

    @lru_cache(maxsize=None)
    def solve(i, busy_mask):
        if i == n:
            return 0
        p, d = jobs[i]
        # Option 1: skip performance i.
        res = solve(i + 1, busy_mask)
        # Option 2: place it in some free slot s in 1..d.
        for s in range(1, d + 1):
            bit = 1 << s
            if not (busy_mask & bit):
                cand = p + solve(i + 1, busy_mask | bit)
                if cand > res:
                    res = cand
        return res

    print(solve(0, 0))

if __name__ == "__main__":
    main()
