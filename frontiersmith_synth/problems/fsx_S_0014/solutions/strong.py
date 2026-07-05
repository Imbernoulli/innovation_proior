# TIER: strong
import sys
from collections import deque

def main():
    d = sys.stdin.buffer.read().split()
    idx = 0
    n = int(d[idx]); idx += 1
    m = int(d[idx]); idx += 1
    r = int(d[idx]); idx += 1
    cost = [0] * (n + 1)
    for v in range(1, n + 1):
        cost[v] = int(d[idx]); idx += 1
    adj = [[] for _ in range(n + 1)]
    for _ in range(m):
        a = int(d[idx]); idx += 1
        b = int(d[idx]); idx += 1
        adj[a].append(b)
        adj[b].append(a)

    ball = [None] * (n + 1)
    for v in range(1, n + 1):
        dist = {v: 0}
        dq = deque([v])
        while dq:
            u = dq.popleft()
            if dist[u] == r:
                continue
            du = dist[u] + 1
            for w in adj[u]:
                if w not in dist:
                    dist[w] = du
                    dq.append(w)
        ball[v] = list(dist.keys())

    # which pumps can serve each tank (for pruning)
    servers = [[] for _ in range(n + 1)]
    for v in range(1, n + 1):
        for w in ball[v]:
            servers[w].append(v)

    def cover_total(chosen):
        return sum(cost[v] for v in chosen)

    def prune(chosen):
        # remove the most expensive pumps that are redundant given the rest
        cov_count = [0] * (n + 1)
        chosen_set = set(chosen)
        for v in chosen:
            for w in ball[v]:
                cov_count[w] += 1
        # try removing in decreasing cost order
        for v in sorted(chosen, key=lambda x: -cost[x]):
            if v not in chosen_set:
                continue
            removable = True
            for w in ball[v]:
                if cov_count[w] <= 1:
                    removable = False
                    break
            if removable:
                chosen_set.discard(v)
                for w in ball[v]:
                    cov_count[w] -= 1
        return list(chosen_set)

    def greedy(key):
        # key(v, gain) -> priority (higher is better). gain = newly-served count.
        covered = bytearray(n + 1)
        remaining = n
        chosen = []
        while remaining > 0:
            best_v = -1
            best_pr = None
            for v in range(1, n + 1):
                gain = 0
                for w in ball[v]:
                    if not covered[w]:
                        gain += 1
                if gain == 0:
                    continue
                pr = key(v, gain)
                if best_pr is None or pr > best_pr:
                    best_pr = pr
                    best_v = v
            if best_v == -1:
                break
            chosen.append(best_v)
            for w in ball[best_v]:
                if not covered[w]:
                    covered[w] = 1
                    remaining -= 1
        for v in range(1, n + 1):
            if not covered[v]:
                chosen.append(v)
                covered[v] = 1
        return chosen

    candidates = []
    # cost-effective greedy: newly-served per unit cost
    candidates.append(greedy(lambda v, gain: gain / float(cost[v])))
    # coverage-first greedy, ties broken by cheaper cost
    candidates.append(greedy(lambda v, gain: (gain, -cost[v])))
    # cost-first-ish: prefer cheap pumps that still serve something
    candidates.append(greedy(lambda v, gain: (gain / float(cost[v]), -cost[v])))

    best = None
    best_val = None
    for ch in candidates:
        pr = prune(ch)
        val = cover_total(pr)
        if best_val is None or val < best_val:
            best_val = val
            best = pr

    # local-search swaps: try replacing a pump by a cheaper single alternative that,
    # together with the rest, still covers its exclusive tanks.
    chosen_set = set(best)
    cov_count = [0] * (n + 1)
    for v in chosen_set:
        for w in ball[v]:
            cov_count[w] += 1
    improved = True
    rounds = 0
    while improved and rounds < 3:
        improved = False
        rounds += 1
        for v in sorted(list(chosen_set), key=lambda x: -cost[x]):
            if v not in chosen_set:
                continue
            # tanks that ONLY v covers right now
            excl = [w for w in ball[v] if cov_count[w] == 1]
            if not excl:
                continue
            # find a single cheaper pump that covers all of excl
            best_alt = None
            best_alt_cost = cost[v]
            # candidate alternates = servers of the first exclusive tank
            for u in servers[excl[0]]:
                if u == v or u in chosen_set:
                    continue
                if cost[u] >= best_alt_cost:
                    continue
                bs = set(ball[u])
                if all(w in bs for w in excl):
                    best_alt = u
                    best_alt_cost = cost[u]
            if best_alt is not None:
                # swap v -> best_alt
                for w in ball[v]:
                    cov_count[w] -= 1
                chosen_set.discard(v)
                chosen_set.add(best_alt)
                for w in ball[best_alt]:
                    cov_count[w] += 1
                improved = True
        # re-prune after swaps
        pr = prune(list(chosen_set))
        chosen_set = set(pr)
        cov_count = [0] * (n + 1)
        for v in chosen_set:
            for w in ball[v]:
                cov_count[w] += 1

    result = list(chosen_set)
    sys.stdout.write("%d\n" % len(result))
    sys.stdout.write(" ".join(str(v) for v in result) + "\n")

main()
