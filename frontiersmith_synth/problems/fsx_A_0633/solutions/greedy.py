# TIER: greedy
"""The obvious first attempt: nearest-neighbor visit the musical rows one at a
time (BFS shortest path each time), with only a thin, non-exact budget
reserve, then try to walk straight home.  This ignores the palindromic-
reflection structure entirely (so it essentially never earns the palindrome
bonus), and -- because it never actually verifies a repeat-free way home
still exists before committing to the next detour -- it can legitimately
strand itself in the Cayley graph on instances where the musical rows are
spread out, failing to close and scoring 0.  That is a real, documented
failure mode of this strategy, not a bug."""
import itertools
import sys
from collections import deque


def apply_call(row, call):
    row = list(row)
    for j in call:
        row[j - 1], row[j] = row[j], row[j - 1]
    return tuple(row)


def identity(n):
    return tuple(range(1, n + 1))


def valid_call(call, n):
    idx = sorted(call)
    if not idx:
        return False
    for a, b in zip(idx, idx[1:]):
        if b - a < 2:
            return False
    return all(1 <= x <= n - 1 for x in idx)


def all_valid_calls(n):
    idxs = list(range(1, n))
    res = []
    for r in range(1, len(idxs) + 1):
        for comb in itertools.combinations(idxs, r):
            if valid_call(comb, n):
                res.append(comb)
    return res


def main():
    data = sys.stdin.read().split("\n")
    header = data[0].split()
    n, Kmax = int(header[0]), int(header[1])
    B = int(data[1].strip())
    targets = []
    for i in range(B):
        toks = data[2 + i].split()
        row = tuple(int(x) for x in toks[1:1 + n])
        targets.append(row)

    calls_all = all_valid_calls(n)
    MAXD = 2 * n + 2

    def bfs_path(src, dst, avoid):
        if src == dst:
            return []
        avoid = avoid - {dst}
        dist = {src: 0}
        prev = {}
        q = deque([src])
        while q:
            r = q.popleft()
            if dist[r] >= MAXD:
                continue
            if r == dst:
                break
            for c in calls_all:
                nr = apply_call(r, c)
                if nr in avoid or nr in dist:
                    continue
                dist[nr] = dist[r] + 1
                prev[nr] = (r, c)
                q.append(nr)
        if dst not in dist:
            return None
        pc = []
        cur = dst
        while cur != src:
            p, c = prev[cur]
            pc.append(c)
            cur = p
        pc.reverse()
        return pc

    RESERVE = 0
    row = identity(n)
    visited = {row}
    calls_out = []
    remaining = [t for t in targets if t != row]
    budget = Kmax
    while remaining and budget - RESERVE > 0:
        best_t = None
        best_path = None
        for t in remaining:
            path = bfs_path(row, t, visited)
            if path is None:
                continue
            if len(path) > budget - RESERVE:
                continue
            if best_path is None or len(path) < len(best_path):
                best_path, best_t = path, t
        if best_t is None:
            break
        for c in best_path:
            row = apply_call(row, c)
            visited.add(row)
            calls_out.append(c)
        budget -= len(best_path)
        remaining.remove(best_t)

    path_home = bfs_path(row, identity(n), visited - {row})
    if path_home is not None:
        for c in path_home:
            row = apply_call(row, c)
            calls_out.append(c)
    if not calls_out:
        calls_out = [(1,), (1,)]

    print(len(calls_out))
    for c in calls_out:
        print(" ".join(map(str, c)))


if __name__ == "__main__":
    main()
