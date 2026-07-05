#!/usr/bin/env python3
# counter.py <in> <out> <ans>  -- deterministic op-count scorer (format D).
#
# Verifies the emitted routing is FUNCTIONALLY EQUIVALENT to the target diagonal
# unitary (every required interaction executed the exact number of times, no more
# no less, each on a hardware edge), THEN counts SWAP operations.  Fewer SWAPs
# => higher ratio.  Baseline B = a naive per-waggle shortest-walk router that the
# checker builds itself (positive).  ratio = min(1000, 100*B/F)/1000.
import sys
from collections import deque, Counter


def read_instance(path):
    data = open(path).read().split()
    idx = 0
    nq = int(data[idx]); ne = int(data[idx + 1]); idx += 2
    adj = [[] for _ in range(nq)]
    edgeset = set()
    for _ in range(ne):
        a = int(data[idx]); b = int(data[idx + 1]); idx += 2
        adj[a].append(b); adj[b].append(a)
        edgeset.add((a, b)); edgeset.add((b, a))
    m = int(data[idx]); idx += 1
    req = []
    for _ in range(m):
        u = int(data[idx]); v = int(data[idx + 1]); idx += 2
        req.append((u, v))
    for a in range(nq):
        adj[a].sort()
    return nq, adj, edgeset, req


def bfs_path(adj, src, dst, nq):
    if src == dst:
        return [src]
    par = [-2] * nq
    par[src] = -1
    q = deque([src])
    while q:
        x = q.popleft()
        for y in adj[x]:
            if par[y] == -2:
                par[y] = x
                if y == dst:
                    path = [dst]
                    c = dst
                    while par[c] != -1:
                        c = par[c]
                        path.append(c)
                    path.reverse()
                    return path
                q.append(y)
    return None


def baseline(nq, adj, req):
    pos = list(range(nq))
    loc = list(range(nq))
    swaps = 0
    for (u, v) in req:
        path = bfs_path(adj, loc[u], loc[v], nq)
        for i in range(len(path) - 2):
            a = path[i]; b = path[i + 1]
            la = pos[a]; lb = pos[b]
            pos[a] = lb; pos[b] = la
            loc[la] = b; loc[lb] = a
            swaps += 1
    return swaps


def fail(reason):
    print("Ratio: 0.0 (%s)" % reason)
    sys.exit(0)


def main():
    inf = sys.argv[1]
    outf = sys.argv[2]
    nq, adj, edgeset, req = read_instance(inf)
    reqc = Counter(frozenset((u, v)) for (u, v) in req)

    pos = list(range(nq))
    execc = Counter()
    swaps = 0
    try:
        lines = open(outf).read().split("\n")
    except Exception:
        fail("no output")

    for ln in lines:
        s = ln.split()
        if not s:
            continue
        op = s[0]
        if op not in ("S", "G") or len(s) < 3:
            fail("malformed instruction line")
        try:
            a = int(s[1]); b = int(s[2])
        except Exception:
            fail("non-integer cell id")
        if a < 0 or b < 0 or a >= nq or b >= nq or a == b:
            fail("cell id out of range or self-loop")
        if (a, b) not in edgeset:
            fail("cells %d,%d do not share a wall (not a hardware edge)" % (a, b))
        if op == "S":
            pos[a], pos[b] = pos[b], pos[a]
            swaps += 1
        else:  # 'G'
            execc[frozenset((pos[a], pos[b]))] += 1

    if execc != reqc:
        fail("interaction multiset does not match the required schedule")

    F = swaps
    B = baseline(nq, adj, req)
    sc = min(1000.0, 100.0 * B / max(1e-9, float(F)))
    print("ops_yours=%d ops_baseline=%d Ratio: %.6f" % (F, B, sc / 1000.0))


if __name__ == "__main__":
    main()
