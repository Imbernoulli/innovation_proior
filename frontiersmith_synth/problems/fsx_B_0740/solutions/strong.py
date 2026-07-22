# TIER: strong
# Insight: makespan is really set by the peak per-sector demand profile, not by raw index
# / critical-path order. Two things follow: (1) whichever sector carries the largest total
# duration is the true bottleneck and must NEVER sit idle once it has ready work, because
# every idle tick there is unrecoverable; (2) within that bottleneck sector, a task should
# be pulled forward exactly in proportion to how much OTHER work is gated behind it, not by
# its position in the input. We precompute, per task, its "unblock weight" W_i (size of its
# descendant set, one reverse pass over the DAG) and, per sector, its total static load.
# The scheduler is an event-driven list scheduler that -- at every decision point -- picks,
# among all currently precedence-ready tasks, the one that can start SOONEST given current
# arm/sector availability; ties are broken first in favour of the HEAVIEST-LOADED sector
# (keep the bottleneck fed), then the largest unblock weight (free the most downstream
# work), then shorter duration. This makes a cheap-but-pivotal "hub" task jump the queue
# ahead of low-value same-sector filler work, while the filler work itself still gets
# continuously drip-fed into the bottleneck sector's idle slots instead of being starved
# until the very end by unrelated, lower-priority parallel work.
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

    children = [[] for _ in range(N + 1)]
    indeg = [0] * (N + 1)
    for (u, v) in edges:
        children[u].append(v)
        indeg[v] += 1

    # unblock weight: size of the descendant closure, via a single reverse (high->low) pass
    # (valid because every edge has u < v, so processing indices N..1 is a reverse-topo order)
    W = [0] * (N + 1)
    for i in range(N, 0, -1):
        tot = 0
        for c in children[i]:
            tot += 1 + W[c]
        W[i] = tot

    # static per-sector total demand -- the peak of this profile is the true bottleneck
    sec_load = [0] * 6
    for i in range(1, N + 1):
        sec_load[s[i - 1]] += d[i - 1]

    earliest_prec = [0] * (N + 1)
    ready_set = [i for i in range(1, N + 1) if indeg[i] == 0]

    arm_free = [0] * K
    sector_free = [0] * 6
    start = [0] * (N + 1)
    out_arm = [0] * (N + 1)
    finish = [0] * (N + 1)
    remaining_indeg = indeg[:]

    remaining = N
    while remaining > 0:
        best_i = None
        best_key = None
        for i in ready_set:
            si = s[i - 1]
            best_arm_free = min(arm_free)
            cand = max(earliest_prec[i], sector_free[si], best_arm_free)
            key = (cand, -sec_load[si], -W[i], d[i - 1], i)
            if best_key is None or key < best_key:
                best_key = key
                best_i = i

        i = best_i
        di = d[i - 1]
        si = s[i - 1]
        best_arm = min(range(K), key=lambda a: arm_free[a])
        cand = max(earliest_prec[i], sector_free[si], arm_free[best_arm])

        start[i] = cand
        out_arm[i] = best_arm
        finish[i] = cand + di
        arm_free[best_arm] = finish[i]
        sector_free[si] = finish[i]

        ready_set.remove(i)
        remaining -= 1

        for c in children[i]:
            if finish[i] > earliest_prec[c]:
                earliest_prec[c] = finish[i]
            remaining_indeg[c] -= 1
            if remaining_indeg[c] == 0:
                ready_set.append(c)

    out = []
    for i in range(1, N + 1):
        out.append("%d %d" % (out_arm[i], start[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
