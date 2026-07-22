# TIER: strong
# Insight: with a fixed-checkpoint schedule, the cost of the mandatory first
# forward pass (1..r1) is unavoidable and identical for every strategy, so all
# that matters is WHICH <= M-1 already-visited nodes you keep resident. That
# is a weighted facility-location problem on the request positions using the
# REAL per-step cost array (read the cost landscape), solved here with an
# O(L^2 * M) DP (L = number of smaller requests). Optimal checkpoints end up
# sitting right where the planted cheap "regenerator" nodes let a checkpoint
# absorb an expensive burst once instead of repaying it for every request
# that follows -- exactly the shape uniform spacing cannot see.
import bisect
import sys


def plan_checkpoints(N, M, reqs, costs):
    prefix = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix[i] = prefix[i - 1] + costs[i]

    q = sorted(reqs)  # ascending
    r1 = q[-1]
    if len(q) <= 1:
        return set(), r1

    items = q[:-1]  # ascending, the K-1 smaller requests
    L = len(items)
    # Reserve 2 live slots for the transient "old predecessor + newly cooked
    # node" pair that is briefly resident together right after each C before
    # the stale predecessor is evicted.
    maxCP = max(0, M - 2)
    pos_prefix = [prefix[p] for p in items]
    S1 = [0.0] * (L + 1)
    for i in range(L):
        S1[i + 1] = S1[i] + pos_prefix[i]

    NEG = float("inf")
    dp = [[NEG] * (maxCP + 1) for _ in range(L + 1)]
    parent = [[None] * (maxCP + 1) for _ in range(L + 1)]
    for j in range(maxCP + 1):
        dp[0][j] = 0.0

    for i in range(1, L + 1):
        for j in range(0, maxCP + 1):
            best, bestp = NEG, None
            for t in range(0, i):
                need = 1 if t > 0 else 0
                jj = j - need
                if jj < 0 or dp[t][jj] == NEG:
                    continue
                anchor_prefix = pos_prefix[t - 1] if t > 0 else 0
                gc = (S1[i] - S1[t]) - (i - t) * anchor_prefix
                val = dp[t][jj] + gc
                if val < best:
                    best, bestp = val, ("take", t)
            dp[i][j] = best
            parent[i][j] = bestp
        for j in range(1, maxCP + 1):
            if dp[i][j - 1] < dp[i][j]:
                dp[i][j] = dp[i][j - 1]
                parent[i][j] = ("skip", j - 1)

    chosen = set()
    i, j = L, maxCP
    while i > 0:
        p = parent[i][j]
        if p is None:
            break
        if p[0] == "skip":
            j = p[1]
            continue
        t = p[1]
        if t > 0:
            chosen.add(items[t - 1])
        i, j = t, (j - 1 if t > 0 else j)

    return chosen, r1


def main():
    data = sys.stdin.read().split("\n")
    N, M, K = (int(x) for x in data[0].split())
    costs_raw = data[1].split()
    costs = [0] * (N + 1)
    for i in range(1, N + 1):
        costs[i] = int(costs_raw[i - 1])
    reqs = [int(x) for x in data[2].split()] if K > 0 else []
    if K == 0:
        return

    chosen, r1 = plan_checkpoints(N, M, reqs, costs)
    cps_sorted = sorted(chosen)

    out = []
    for i in range(1, r1 + 1):
        out.append("C %d" % i)
        if i > 1 and (i - 1) not in chosen:
            out.append("E %d" % (i - 1))
    out.append("U %d" % r1)
    if r1 not in chosen:
        out.append("E %d" % r1)

    for r in reqs[1:]:
        idx = bisect.bisect_right(cps_sorted, r) - 1
        cp = cps_sorted[idx] if idx >= 0 else 0
        for i in range(cp + 1, r + 1):
            out.append("C %d" % i)
            if i > cp + 1 and (i - 1) not in chosen:
                out.append("E %d" % (i - 1))
        out.append("U %d" % r)
        if r not in chosen:
            out.append("E %d" % r)

    sys.stdout.write("\n".join(out) + ("\n" if out else ""))


if __name__ == "__main__":
    main()
