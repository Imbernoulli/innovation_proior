# TIER: greedy
# Textbook "uniform sqrt-checkpointing" recipe: space checkpoints evenly across
# [1, r1] purely by arithmetic (position only), completely ignoring the actual
# per-step cost array.  This is the obvious first idea for anyone who knows the
# classic recompute-checkpointing trick -- it is blind to planted cost bursts.
import bisect
import sys


def main():
    data = sys.stdin.read().split("\n")
    N, M, K = (int(x) for x in data[0].split())
    reqs = [int(x) for x in data[2].split()] if K > 0 else []
    if K == 0:
        return

    r1 = reqs[0]
    # Reserve 2 live slots for the transient "old predecessor + newly cooked
    # node" pair that is briefly resident together right after each C before
    # the stale predecessor is evicted.
    S = max(0, M - 2)
    S = min(S, max(0, r1 - 1))

    cp_set = set()
    if S > 0:
        for i in range(1, S + 1):
            p = round(i * r1 / (S + 1))
            p = max(1, min(r1, p))
            cp_set.add(p)
    cps_sorted = sorted(cp_set)

    out = []
    # Initial forward pass 1..r1, keeping only checkpoint positions resident.
    for i in range(1, r1 + 1):
        out.append("C %d" % i)
        if i > 1 and (i - 1) not in cp_set:
            out.append("E %d" % (i - 1))
    out.append("U %d" % r1)
    if r1 not in cp_set:
        out.append("E %d" % r1)

    for r in reqs[1:]:
        idx = bisect.bisect_right(cps_sorted, r) - 1
        cp = cps_sorted[idx] if idx >= 0 else 0
        for i in range(cp + 1, r + 1):
            out.append("C %d" % i)
            if i > cp + 1 and (i - 1) not in cp_set:
                out.append("E %d" % (i - 1))
        out.append("U %d" % r)
        if r not in cp_set:
            out.append("E %d" % r)

    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
