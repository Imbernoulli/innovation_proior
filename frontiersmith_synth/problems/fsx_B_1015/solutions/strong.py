# TIER: strong
# INSIGHT: because the walk is self-avoiding and irreversible, every branch
# taken is a commitment -- stepping toward the locally richest neighbour
# ignores whether that step *preserves an escape frontier* to the rest of the
# map.  Instead of ranking neighbours by their own reward, evaluate each
# branch by the TOTAL reward reachable if you commit to it (a full lookahead
# search), and track which keys are held so a gate is only crossed once its
# key has actually been collected earlier on THIS path.  Because every decoy
# spur is a strict dead end (bounded, no further branching) and the only real
# fork is the small key/no-key diamond, the decision tree is linear in the
# map size, not exponential, so an exhaustive search is cheap and exact: it
# always skips every decoy (their total is dwarfed by the vault), always
# takes the key branch, and rides the gate into the vault tail.
import sys


def main():
    sys.setrecursionlimit(10000)
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
    path = [S]
    best_total = [cells[S][2]]
    best_path = [list(path)]

    def rec(cur, total):
        if total > best_total[0]:
            best_total[0] = total
            best_path[0] = list(path)
        for v in sorted(adj[cur]):
            if visited[v]:
                continue
            _, _, rew, k, kid = cells[v]
            if k == 2 and kid not in keys:
                continue
            visited[v] = True
            added_key = None
            if k == 1 and kid not in keys:
                keys.add(kid)
                added_key = kid
            path.append(v)
            rec(v, total + rew)
            path.pop()
            if added_key is not None:
                keys.discard(added_key)
            visited[v] = False

    rec(S, cells[S][2])

    out = [str(len(best_path[0]))]
    for u in best_path[0]:
        out.append("%d %d" % (cells[u][0], cells[u][1]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
