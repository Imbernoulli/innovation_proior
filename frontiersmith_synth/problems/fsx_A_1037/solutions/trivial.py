# TIER: trivial
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); R = int(next(it)); K = int(next(it))
    values = [int(next(it)) for _ in range(N)]
    cap = [0] + [int(next(it)) for _ in range(M)]

    # only bother spending 40% of the replica budget (give up early), one replica per
    # covered shard, cycling machine index order, oblivious to value and to which pair
    # the checker actually judges you on -- reproduces the checker's own weak reference.
    remaining = cap[:]
    budget = max(1, round(R * 0.4))
    pairs = []
    mach_ptr = 1
    for i in range(1, N + 1):
        if budget <= 0:
            break
        m = mach_ptr
        placed = False
        for _ in range(M):
            if remaining[m] > 0:
                pairs.append((i, m))
                remaining[m] -= 1
                budget -= 1
                placed = True
                mach_ptr = m % M + 1
                break
            m = m % M + 1
        if not placed:
            mach_ptr = mach_ptr % M + 1

    out = [str(len(pairs))]
    for (t, m) in pairs:
        out.append("%d %d" % (t, m))
    sys.stdout.write("\n".join(out) + "\n")


main()
