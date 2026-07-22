# TIER: greedy
import sys


def main():
    toks = sys.stdin.read().split()
    it = iter(toks)
    N = int(next(it)); M = int(next(it)); R = int(next(it)); K = int(next(it))
    values = [int(next(it)) for _ in range(N)]
    cap = [0] + [int(next(it)) for _ in range(M)]

    # obvious textbook approach: give the highest-value shards up to two replicas each,
    # filling machines in plain increasing-index order (first machine with spare room).
    # This is the "protect everyone a little, everywhere, evenly" minimax instinct --
    # it never asks which disaster is safe to write off, so it burns budget fragmenting
    # coverage across many small machines instead of concentrating it.
    order = sorted(range(1, N + 1), key=lambda i: -values[i - 1])

    remaining = cap[:]
    budget = R
    pairs = []
    used = [None] + [set() for _ in range(N)]

    TARGET_REPLICAS = 2
    for _rep in range(TARGET_REPLICAS):
        if budget <= 0:
            break
        for t in order:
            if budget <= 0:
                break
            for m in range(1, M + 1):
                if remaining[m] > 0 and m not in used[t]:
                    used[t].add(m)
                    remaining[m] -= 1
                    budget -= 1
                    pairs.append((t, m))
                    break

    out = [str(len(pairs))]
    for (t, m) in pairs:
        out.append("%d %d" % (t, m))
    sys.stdout.write("\n".join(out) + "\n")


main()
