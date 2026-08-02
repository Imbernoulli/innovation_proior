# TIER: greedy
# The obvious first move: the FIRST visible grid is given in full, so just
# BFS the shortest path on its raw coordinates (walls = '#', everything
# else = floor) and hardcode the exact move sequence as a chain of
# one-shot states: state i, whatever token it happens to see there -> move
# i, advance to state i+1.  This perfectly replays the ONE grid it was
# computed on (it needed the exact same number of moves as the shortest
# path, so it also looks efficient there) -- but the token seen at state i
# on any OTHER grid is essentially never the token this chain expects, so
# it falls back to WAIT almost immediately and stalls forever: it never
# even reaches the relay beacon on the other 9 grids.  It also burns far
# more states/rules than the reactive rule needs, paying the size penalty
# too.
import sys, json
from collections import deque

inst = json.load(sys.stdin)
g0 = inst["visible_grids"][0]
H, W = g0["height"], g0["width"]
grid = g0["grid"]
sr, sc = g0["start"]

goal = None
for r in range(H):
    for c in range(W):
        if grid[r][c] == "X":
            goal = (r, c)

DELTA = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}

dist = {(sr, sc): 0}
par = {}
q = deque([(sr, sc)])
while q:
    r, c = q.popleft()
    if (r, c) == goal:
        break
    for d, (dr, dc) in DELTA.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != "#" and (nr, nc) not in dist:
            dist[(nr, nc)] = dist[(r, c)] + 1
            par[(nr, nc)] = ((r, c), d)
            q.append((nr, nc))

acts = []
cur = goal
while cur != (sr, sc):
    prev, d = par[cur]
    acts.append(d)
    cur = prev
acts.reverse()

rules = []
r, c = sr, sc
for i, a in enumerate(acts):
    tok = grid[r][c]
    rules.append({"state": i, "see": tok, "action": a, "next": i + 1})
    dr, dc = DELTA[a]
    r, c = r + dr, c + dc

answer = {"start_state": 0, "rules": rules}
print(json.dumps(answer))
