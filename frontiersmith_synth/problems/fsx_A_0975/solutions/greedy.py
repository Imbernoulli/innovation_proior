# TIER: greedy
# The universal CAM baseline: cut each part's contour with its own pierce, in
# the order parts are listed. To respect "cut each required edge exactly
# once", a shared edge is claimed by whichever part reaches it first; the
# remaining un-claimed edges of a part always form one connected path (a
# rectangle missing 0-2 adjacent sides is still a path/cycle), so a single
# continuous trail per part suffices. No attempt is made to reduce the
# number of pierces below "one per part", and parts are visited in raw
# input order (no reordering for shorter dead travel).
import sys

def walk(edge_ids, edge_uv):
    adj = {}
    for e in edge_ids:
        u, v = edge_uv[e]
        adj.setdefault(u, []).append((v, e))
        adj.setdefault(v, []).append((u, e))
    deg = {v: len(l) for v, l in adj.items()}
    start = None
    for v, d in deg.items():
        if d % 2 == 1:
            start = v
            break
    if start is None:
        start = next(iter(adj))
    used = set()
    path = [start]
    cur = start
    while True:
        nxt = None
        for v, e in adj[cur]:
            if e not in used:
                nxt = (v, e)
                break
        if nxt is None:
            break
        v, e = nxt
        used.add(e)
        path.append(v)
        cur = v
    return path, used

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    def rdi(): return int(next(it))
    n = rdi(); m = rdi(); P = rdi()
    for _ in range(n):
        rdi(); rdi()
    edge_uv = {0: None}
    for e in range(1, m + 1):
        u = rdi(); v = rdi()
        edge_uv[e] = (u, v)
    K = rdi()
    parts = []
    for _ in range(K):
        a = rdi(); b = rdi(); c = rdi(); d = rdi()
        parts.append([a, b, c, d])

    claimed = set()
    trails = []
    for part in parts:
        remain = [e for e in part if e not in claimed]
        if not remain:
            continue
        path, used_here = walk(remain, edge_uv)
        claimed |= used_here
        trails.append(path)

    out = [str(len(trails))]
    for path in trails:
        out.append(f"{len(path)-1} " + " ".join(map(str, path)))
    sys.stdout.write("\n".join(out) + "\n")

if __name__ == "__main__":
    main()
