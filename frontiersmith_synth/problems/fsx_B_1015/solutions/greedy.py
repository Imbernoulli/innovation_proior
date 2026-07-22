# TIER: greedy
# The obvious first attempt: at every step, walk to whichever legal unvisited
# neighbour has the biggest immediate reward (one-step lookahead, no memory
# of what lies further ahead).  This is exactly the trap: the very first
# decoy spur dangles a fat reward on its entrance cell, greedy dives in, and
# the spur is a strict dead end -- the walk gets permanently stuck there,
# forfeiting the entire rest of the spine, the key, the gate and the vault.
import sys


def main():
    it = sys.stdin.read().split()
    p = 0
    N = int(it[p]); V = int(it[p + 1]); E = int(it[p + 2]); S = int(it[p + 3]); p += 4
    cells = []
    for i in range(V):
        r = int(it[p]); c = int(it[p + 1]); rew = int(it[p + 2])
        k = int(it[p + 3]); kid = int(it[p + 4]); p += 5
        cells.append((r, c, rew, k, kid))
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
        best = None; bestrew = -1
        for v in sorted(adj[cur]):
            if visited[v]:
                continue
            _, _, rew, k, kid = cells[v]
            if k == 2 and kid not in keys:
                continue
            if rew > bestrew:
                bestrew = rew; best = v
        if best is None:
            break
        visited[best] = True
        if cells[best][3] == 1:
            keys.add(cells[best][4])
        cur = best
        path.append(cur)

    out = [str(len(path))]
    for u in path:
        out.append("%d %d" % (cells[u][0], cells[u][1]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
