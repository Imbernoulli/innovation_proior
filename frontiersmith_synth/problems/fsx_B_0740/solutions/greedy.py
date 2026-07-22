# TIER: greedy
# The "obvious" approach: forward ASAP list scheduling in fixed input (topological) order.
# For each task, in index order, commit it to whichever arm frees up earliest, at the
# earliest tick allowed by its precedence predecessors (already committed) and its sector's
# mutex queue. This never reorders tasks, so on planted instances it drains a heavy sector's
# low-value filler tasks before a late-indexed but high-value "hub" task, needlessly
# stalling everything that depends on the hub while other arms sit idle.
import sys


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    d = [int(next(it)) for _ in range(N)]
    s = [int(next(it)) for _ in range(N)]
    E = int(next(it))
    edges = []
    for _ in range(E):
        u = int(next(it)); v = int(next(it))
        edges.append((u, v))

    preds = [[] for _ in range(N + 1)]
    for (u, v) in edges:
        preds[v].append(u)

    arm_free = [0] * K
    sector_free = [0] * 6
    start = [0] * (N + 1)
    finish = [0] * (N + 1)

    out = []
    for i in range(1, N + 1):
        di = d[i - 1]
        si = s[i - 1]
        earliest_prec = 0
        for u in preds[i]:
            fu = finish[u]
            if fu > earliest_prec:
                earliest_prec = fu
        # pick the arm that frees up soonest
        best_arm = min(range(K), key=lambda a: arm_free[a])
        cand = max(earliest_prec, sector_free[si], arm_free[best_arm])
        start[i] = cand
        finish[i] = cand + di
        arm_free[best_arm] = finish[i]
        sector_free[si] = finish[i]
        out.append("%d %d" % (best_arm, cand))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
