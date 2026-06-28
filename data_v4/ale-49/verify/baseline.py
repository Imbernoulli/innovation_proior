#!/usr/bin/env python3
"""Trivial baseline solver for ale-49: strictly sequential one-at-a-time mover.

Reads an instance on stdin, writes a feasible action sequence on stdout.

Strategy (deliberately simple / weak, to serve as the reference baseline):
  Route tokens ONE AT A TIME in input order.  To move the active token, BFS to its
  goal treating every other token as a wall.  If blocked, relocate the first
  blocking token to any reachable free cell off the route, then retry.  No
  parallelism, no priority ordering, no space-time reasoning -- so its action
  count is typically much larger than the prioritized planner's.
"""
import sys
from collections import deque

DR = [-1, 1, 0, 0]
DC = [0, 0, -1, 1]
DCH = ['U', 'D', 'L', 'R']


def main():
    data = sys.stdin.read().split('\n')
    idx = 0
    while data[idx].strip() == '':
        idx += 1
    H, W, K = map(int, data[idx].split()); idx += 1
    grid = []
    for _ in range(H):
        row = data[idx]; idx += 1
        if len(row) < W:
            row = row + '.' * (W - len(row))
        grid.append(row[:W])
    starts = []; targets = []
    for _ in range(K):
        while data[idx].strip() == '':
            idx += 1
        sr, sc, tr, tc = map(int, data[idx].split()); idx += 1
        starts.append(sr * W + sc); targets.append(tr * W + tc)

    def freec(cell):
        r, c = divmod(cell, W)
        return 0 <= r < H and 0 <= c < W and grid[r][c] != '#'

    cur = starts[:]
    actions = []

    def occupied_by(cell, self):
        for j in range(K):
            if j != self and cur[j] == cell:
                return j
        return -1

    def bfs(self, src, dst, treat_tokens_as_walls):
        prev = [-2] * (H * W)
        blocked = [False] * (H * W)
        if treat_tokens_as_walls:
            for j in range(K):
                if j != self:
                    blocked[cur[j]] = True
        prev[src] = -1
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                break
            r, c = divmod(u, W)
            for d in range(4):
                nr, nc = r + DR[d], c + DC[d]
                if not (0 <= nr < H and 0 <= nc < W) or grid[nr][nc] == '#':
                    continue
                v = nr * W + nc
                if blocked[v] and v != dst:
                    continue
                if prev[v] != -2:
                    continue
                prev[v] = u
                q.append(v)
        if prev[dst] == -2:
            return None
        path = []
        x = dst
        while x != -1:
            path.append(x); x = prev[x]
        path.reverse()
        return path

    def dir_between(a, b):
        ar, ac = divmod(a, W); br, bc = divmod(b, W)
        for d in range(4):
            if ar + DR[d] == br and ac + DC[d] == bc:
                return DCH[d]
        return '?'

    def move_along(self, path):
        for s in range(1, len(path)):
            actions.append((self, dir_between(path[s - 1], path[s])))
            cur[self] = path[s]

    def relocate(b, forbidden):
        prev = [-2] * (H * W)
        blocked = [False] * (H * W)
        for j in range(K):
            if j != b:
                blocked[cur[j]] = True
        prev[cur[b]] = -1
        q = deque([cur[b]])
        dest = -1
        while q:
            u = q.popleft()
            if u != cur[b] and u not in forbidden:
                dest = u; break
            r, c = divmod(u, W)
            for d in range(4):
                nr, nc = r + DR[d], c + DC[d]
                if not (0 <= nr < H and 0 <= nc < W) or grid[nr][nc] == '#':
                    continue
                v = nr * W + nc
                if blocked[v] or prev[v] != -2:
                    continue
                prev[v] = u; q.append(v)
        if dest == -1:
            return False
        path = []; x = dest
        while x != -1:
            path.append(x); x = prev[x]
        path.reverse()
        move_along(b, path)
        return True

    for i in range(K):
        goal = targets[i]
        guard = 0
        while cur[i] != goal:
            guard += 1
            if guard > 8 * H * W + 64:
                break
            path = bfs(i, cur[i], goal, True)
            if path is not None:
                move_along(i, path)
                break
            relaxed = bfs(i, cur[i], goal, False)
            if relaxed is None:
                break
            blocker = -1
            for cell in relaxed:
                b = occupied_by(cell, i)
                if b != -1:
                    blocker = b; break
            if blocker == -1:
                break
            forbidden = set(relaxed); forbidden.add(goal)
            if not relocate(blocker, forbidden):
                break

    out = [str(len(actions))]
    for (i, d) in actions:
        out.append(f"{i} {d}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
