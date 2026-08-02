# TIER: greedy
"""The obvious first attempt: treat this as a pure load-balancing / bin-
packing problem. Sort keys by weight descending (LPT -- Longest Processing
Time first, the textbook multiprocessor-scheduling heuristic) and drop each
key into whichever shard currently has the least total weight. This makes
shard sizes as uniform as possible -- it minimizes the size-skew term -- but
it has NO notion of which keys transact together and NO notion of the
previous epoch's placement, so it maximizes cross-shard transaction cost on
any workload with real co-transaction structure, and pays needless migration
cost whenever a perfectly good previous assignment already existed."""
import sys


def main():
    data = sys.stdin.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        v = data[pos]
        pos += 1
        return v

    n = int(nxt())
    K = int(nxt())
    nxt(); nxt(); nxt()  # A B G -- unused by this heuristic
    weights = [int(nxt()) for _ in range(n)]
    # prev assignment and transaction edges are read but deliberately ignored
    for _ in range(n):
        nxt()
    m = int(nxt())
    for _ in range(m):
        nxt(); nxt(); nxt()

    order = sorted(range(n), key=lambda i: (-weights[i], i))
    loads = [0] * K
    assign = [0] * n
    for i in order:
        s = min(range(K), key=lambda s: (loads[s], s))
        assign[i] = s
        loads[s] += weights[i]

    print(" ".join(str(a) for a in assign))


if __name__ == "__main__":
    main()
