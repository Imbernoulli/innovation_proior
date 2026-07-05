# TIER: strong
# No reset + meet-in-the-middle placement with look-ahead.  For each splice the two
# channels are brought adjacent along a shortest path; the SPLIT point (how far each
# channel travels) costs the same d-1 swaps but leaves the pair in different final
# boxes.  We pick the split that minimises the summed distance of the next W splices'
# channel pairs -- steering channels toward where they will next be needed.
import sys
from collections import deque

W = 4  # look-ahead window


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it)); e = int(next(it)); q = int(next(it))
    adj = [[] for _ in range(n)]
    for _ in range(e):
        u = int(next(it)); v = int(next(it))
        adj[u].append(v); adj[v].append(u)
    placement = [int(next(it)) for _ in range(n)]
    ops = [(int(next(it)), int(next(it))) for _ in range(q)]

    # all-pairs shortest path (n is small: <= ~110)
    dist = []
    for src in range(n):
        dd = [-1] * n
        dd[src] = 0
        dq = deque([src])
        while dq:
            x = dq.popleft()
            for y in adj[x]:
                if dd[y] < 0:
                    dd[y] = dd[x] + 1
                    dq.append(y)
        dist.append(dd)

    token_at = list(placement)
    site_at = [0] * n
    for s, t in enumerate(placement):
        site_at[t] = s

    def bfs_path(src, dst):
        prev = [-2] * n
        prev[src] = -1
        dq = deque([src])
        while dq:
            x = dq.popleft()
            if x == dst:
                break
            for y in adj[x]:
                if prev[y] == -2:
                    prev[y] = x
                    dq.append(y)
        path = []
        x = dst
        while x != -1:
            path.append(x)
            x = prev[x]
        path.reverse()
        return path

    def do_swap(u, v, out):
        tu = token_at[u]; tv = token_at[v]
        token_at[u] = tv; token_at[v] = tu
        site_at[tu] = v; site_at[tv] = u
        out.append("S %d %d" % (u, v))

    out = []
    for k in range(q):
        a, b = ops[k]
        sa = site_at[a]; sb = site_at[b]
        path = bfs_path(sa, sb)  # nodes sa=path[0] .. sb=path[-1]
        d = len(path) - 1
        if d <= 1:
            out.append("G %d" % (k + 1))
            continue

        # future pairs (channel ids) in the look-ahead window
        future = ops[k + 1:k + 1 + W]

        # candidate split s in 0..d-1: channel a ends at path[s], channel b ends at path[s+1]
        best_s = 0
        best_cost = None
        for s in range(d):
            fa = path[s]        # future box of channel a
            fb = path[s + 1]    # future box of channel b
            cost = 0
            for (x, y) in future:
                # approximate: only a and b move; use their prospective boxes
                sx = fa if x == a else (fb if x == b else site_at[x])
                sy = fa if y == a else (fb if y == b else site_at[y])
                cost += dist[sx][sy]
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_s = s

        s = best_s
        # realise: move channel a forward s steps (path[0]->path[s])
        for j in range(s):
            do_swap(path[j], path[j + 1], out)
        # move channel b backward to path[s+1] (it currently sits at path[d])
        for j in range(d, s + 1, -1):
            do_swap(path[j], path[j - 1], out)
        out.append("G %d" % (k + 1))

    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
