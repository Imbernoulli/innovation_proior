# TIER: greedy
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

    # ball[v] = tanks served by a pump at v (within hop distance r)
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

    # cost-BLIND max-coverage greedy: repeatedly install where the most currently-
    # unserved tanks become served. Ignores cost -> tends to grab expensive hubs.
    covered = bytearray(n + 1)
    remaining = n
    chosen = []
    while remaining > 0:
        best_v = -1
        best_gain = -1
        for v in range(1, n + 1):
            if covered[v] and False:
                pass
            gain = 0
            for w in ball[v]:
                if not covered[w]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_v = v
        if best_v == -1 or best_gain == 0:
            break
        chosen.append(best_v)
        for w in ball[best_v]:
            if not covered[w]:
                covered[w] = 1
                remaining -= 1

    # safety: cover anything still uncovered by its own pump
    for v in range(1, n + 1):
        if not covered[v]:
            chosen.append(v)
            covered[v] = 1

    # dedup preserve order
    seen = set()
    uniq = []
    for v in chosen:
        if v not in seen:
            seen.add(v)
            uniq.append(v)

    sys.stdout.write("%d\n" % len(uniq))
    sys.stdout.write(" ".join(str(v) for v in uniq) + "\n")

main()
