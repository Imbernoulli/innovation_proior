import sys
from collections import deque

def fail(msg):
    print("Ratio: 0.0 (%s)" % msg)
    sys.exit(0)

def main():
    inp = open(sys.argv[1]).read().split()
    out = open(sys.argv[2]).read().split()

    it = iter(inp)
    def ni():
        return int(next(it))

    try:
        n = ni(); m = ni(); r = ni()
    except Exception:
        fail("bad header")

    cost = [0] * (n + 1)
    try:
        for v in range(1, n + 1):
            cost[v] = ni()
    except Exception:
        fail("bad costs")

    adj = [[] for _ in range(n + 1)]
    try:
        for _ in range(m):
            a = ni(); b = ni()
            adj[a].append(b)
            adj[b].append(a)
    except Exception:
        fail("bad edges")

    B = sum(cost[1:n + 1])
    B = max(1, B)

    # ---- parse participant output ----
    if len(out) == 0:
        fail("empty output")
    try:
        s = int(out[0])
    except Exception:
        fail("s not an integer")
    if s < 0 or s > n:
        fail("s out of range: %d" % s)
    if len(out) != 1 + s:
        fail("expected %d indices after s=%d, got %d tokens total" % (s, s, len(out)))

    picks = []
    seen = [False] * (n + 1)
    for j in range(s):
        try:
            p = int(out[1 + j])
        except Exception:
            fail("index not an integer: %r" % out[1 + j])
        if p < 1 or p > n:
            fail("index out of range: %d" % p)
        if seen[p]:
            fail("duplicate index: %d" % p)
        seen[p] = True
        picks.append(p)

    # ---- coverage: multi-source BFS from picks, depth-limited to r ----
    covered = [False] * (n + 1)
    dist = [-1] * (n + 1)
    dq = deque()
    for p in picks:
        if dist[p] == -1:
            dist[p] = 0
            covered[p] = True
            dq.append(p)
    while dq:
        u = dq.popleft()
        if dist[u] == r:
            continue
        du = dist[u] + 1
        for w in adj[u]:
            if dist[w] == -1:
                dist[w] = du
                covered[w] = True
                dq.append(w)
    for v in range(1, n + 1):
        if not covered[v]:
            fail("tank %d not served within radius %d" % (v, r))

    # ---- objective ----
    F = sum(cost[p] for p in picks)
    F = max(1, F)

    sc = min(1000.0, 100.0 * float(B) / float(F))
    print("F=%d B=%d s=%d Ratio: %.6f" % (F, B, s, sc / 1000.0))

if __name__ == "__main__":
    main()
