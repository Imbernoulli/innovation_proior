# TIER: greedy
# The obvious first pass: a purely reactive damage-hunter. If a unit cannot
# reach anyone to attack this turn it holds still (so this isn't a strawman
# that wanders for no reason when there's nothing to do) -- but the INSTANT
# some enemy is reachable this turn, the unit beelines for whichever enemy
# currently has the globally lowest HP and attacks it, with zero regard for
# terrain, objective tiles, or zone-of-control on the way there. It never
# comes back for ground it abandoned mid-chase, and it never coordinates
# focus fire across units beyond "whoever is weakest right now" -- so two
# units can easily converge on the same easy kill while a wounded ally that
# would have died to one more hit gets ignored because another unit farther
# away looks like an even easier kill this instant. This maximizes
# immediate kills, not the position that determines the next few turns.
import sys, json
from collections import deque


def in_bounds(x, y, W, H):
    return 0 <= x < W and 0 <= y < H


def neighbors4(x, y):
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def is_zoc(tile, opp_pos_set):
    x, y = tile
    for nb in neighbors4(x, y):
        if nb in opp_pos_set:
            return True
    return False


def reachable(start, move_pts, occupied, opp_pos_set, terrain, W, H):
    dist = {start: 0}
    result = {start}
    dq = deque([start])
    while dq:
        cur = dq.popleft()
        if cur != start and is_zoc(cur, opp_pos_set):
            continue
        if dist[cur] >= move_pts:
            continue
        cx, cy = cur
        for nb in neighbors4(cx, cy):
            if nb in dist:
                continue
            nx, ny = nb
            if not in_bounds(nx, ny, W, H):
                continue
            if terrain[ny][nx] < 0:
                continue
            if nb in occupied:
                continue
            dist[nb] = dist[cur] + 1
            result.add(nb)
            dq.append(nb)
    return result


inst = json.load(sys.stdin)
W, H = inst["W"], inst["H"]
terrain = inst["terrain"]
friendly = {u["id"]: dict(u) for u in inst["friendly"]}
enemy = {u["id"]: dict(u) for u in inst["enemy"]}
order = [u["id"] for u in inst["friendly"]]
occupied_f = set((u["x"], u["y"]) for u in inst["friendly"])

orders_out = []
for fid in order:
    u = friendly[fid]
    start = (u["x"], u["y"])
    move_to = None
    atk = None
    if enemy:
        enemy_positions = set((e["x"], e["y"]) for e in enemy.values())
        occ = (occupied_f - {start}) | enemy_positions
        reach = reachable(start, u["move"], occ, enemy_positions, terrain, W, H)
        best_key = None
        best_tile = None
        best_eid = None
        for t in reach:
            adj = [eid for eid, e in enemy.items() if abs(e["x"] - t[0]) + abs(e["y"] - t[1]) == 1]
            if not adj:
                continue
            weakest = min(adj, key=lambda eid: (enemy[eid]["hp"], eid))
            steps = abs(t[0] - start[0]) + abs(t[1] - start[1])
            key = (enemy[weakest]["hp"], weakest, steps, t)
            if best_key is None or key < best_key:
                best_key = key
                best_tile = t
                best_eid = weakest
        if best_tile is not None:
            move_to = best_tile
            atk = best_eid
    if move_to is not None:
        occupied_f.discard(start)
        occupied_f.add(move_to)
        u["x"], u["y"] = move_to
    orders_out.append({
        "unit_id": fid,
        "move_to": (None if move_to is None or move_to == start else list(move_to)),
        "attack": atk,
    })

print(json.dumps({"orders": orders_out, "state": None}))
