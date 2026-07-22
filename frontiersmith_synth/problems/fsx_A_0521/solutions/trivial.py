# TIER: trivial
# Independent shortest paths, every courier goes ASAP, all solo.
# Reproduces the checker's baseline B  ->  ratio ~= 0.1.
import sys, heapq


def main():
    dat = sys.stdin.read().split()
    p = 0
    N = int(dat[p]); p += 1
    M = int(dat[p]); p += 1
    K = int(dat[p]); p += 1
    p += 2  # alpha beta
    adj = [[] for _ in range(N)]
    for _ in range(M):
        u = int(dat[p]); v = int(dat[p + 1]); w = int(dat[p + 2]); p += 3
        adj[u].append((v, w)); adj[v].append((u, w))
    cour = []
    for _ in range(K):
        s = int(dat[p]); g = int(dat[p + 1]); d = int(dat[p + 2]); p += 3
        cour.append((s, g, d))

    def shortest_path(src, dst):
        dist = [None] * N
        par = [-1] * N
        dist[src] = 0
        pq = [(0, src)]
        while pq:
            dd, u = heapq.heappop(pq)
            if dd > dist[u]:
                continue
            for (v, w) in adj[u]:
                nd = dd + w
                if dist[v] is None or nd < dist[v]:
                    dist[v] = nd; par[v] = u
                    heapq.heappush(pq, (nd, v))
        path = []
        cur = dst
        while cur != -1:
            path.append(cur); cur = par[cur]
        path.reverse()
        return path

    lines = []
    for i in range(K):
        s, g, d = cour[i]
        path = shortest_path(s, g)
        toks = [str(i), str(len(path) - 1)]
        for h in range(len(path) - 1):
            toks.append("L"); toks.append(str(path[h + 1]))
        lines.append(" ".join(toks))
    sys.stdout.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
