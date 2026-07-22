# TIER: trivial
# Baseline construction: a canonical, reward-BLIND direction-priority walker
# (always prefer right, then down, then up, then left among legal moves).  It
# reproduces the checker's own internal baseline B exactly, so it scores
# Ratio ~= 0.1 on every case.  It happens to follow the spine (since 'right'
# beats 'down', it skips every decoy spur) but at the fork it also prefers
# 'right' -- the branch WITHOUT the key -- so it starves at the locked gate
# and never reaches the vault.
import sys

DIRS = [(0, 1), (1, 0), (-1, 0), (0, -1)]


def main():
    it = sys.stdin.read().split()
    p = 0
    N = int(it[p]); V = int(it[p + 1]); E = int(it[p + 2]); S = int(it[p + 3]); p += 4
    cells = []
    pos2idx = {}
    for i in range(V):
        r = int(it[p]); c = int(it[p + 1]); rew = int(it[p + 2])
        k = int(it[p + 3]); kid = int(it[p + 4]); p += 5
        cells.append((r, c, rew, k, kid))
        pos2idx[(r, c)] = i
    adj = [set() for _ in range(V)]
    for _ in range(E):
        u = int(it[p]); v = int(it[p + 1]); p += 2
        adj[u].add(v); adj[v].add(u)

    visited = [False] * V
    visited[S] = True
    keys = set()
    cur = S
    path = [S]
    while True:
        r, c = cells[cur][0], cells[cur][1]
        moved = False
        for dr, dc in DIRS:
            v = pos2idx.get((r + dr, c + dc))
            if v is None or v not in adj[cur] or visited[v]:
                continue
            _, _, rew, k, kid = cells[v]
            if k == 2 and kid not in keys:
                continue
            visited[v] = True
            if k == 1:
                keys.add(kid)
            cur = v
            path.append(v)
            moved = True
            break
        if not moved:
            break

    out = [str(len(path))]
    for u in path:
        out.append("%d %d" % (cells[u][0], cells[u][1]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
