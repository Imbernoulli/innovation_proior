# TIER: strong
# The insight: value a reachable tile by the defense bonus it gives you and
# denies the enemy for the turns that follow, not by the damage a swing
# from it deals this instant. Every reachable tile (BFS, respecting
# zone-of-control) is scored terrain_defense*2 + 5 if it is one of the two
# disclosed objective tiles; only an attack opportunity that FINISHES a kill
# this turn (tracked via pending_dmg, coordinating focus fire across the
# units already processed this same turn -- see the innovation addendum's
# "focus-fire-vs-spread") is allowed to outweigh good ground, because a kill
# denies the target's own attack AND its tile-denial for the rest of the
# game, while a non-lethal poke is worth only a small, tie-breaking nudge
# toward pressuring whichever enemy currently squats on the best terrain.
# Absent any such opportunity the unit simply holds the best tile it can
# already reach (frequently its own current tile) and lets the fixed enemy
# AI walk into a fight that costs it more than it costs us.
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


def dmg(atk, defb):
    return max(1, atk - max(0, defb))


inst = json.load(sys.stdin)
W, H = inst["W"], inst["H"]
terrain = inst["terrain"]
obj = set(tuple(t) for t in inst["objective_tiles"])
friendly = {u["id"]: dict(u) for u in inst["friendly"]}
enemy = {u["id"]: dict(u) for u in inst["enemy"]}
order = [u["id"] for u in inst["friendly"]]
occupied_f = set((u["x"], u["y"]) for u in inst["friendly"])
pending_dmg = {}

orders_out = []
for fid in order:
    u = friendly[fid]
    start = (u["x"], u["y"])
    enemy_positions = set((e["x"], e["y"]) for e in enemy.values() if e["hp"] > 0)
    occ = (occupied_f - {start}) | enemy_positions
    reach = reachable(start, u["move"], occ, enemy_positions, terrain, W, H)

    def tile_value(t):
        tx, ty = t
        v = terrain[ty][tx] * 2.0
        if t in obj:
            v += 5.0
        return v

    def combat_bonus(t):
        tx, ty = t
        best = 0.0
        for eid, e in enemy.items():
            if e["hp"] <= 0:
                continue
            if abs(e["x"] - tx) + abs(e["y"] - ty) != 1:
                continue
            d = dmg(u["atk"], terrain[e["y"]][e["x"]])
            already = pending_dmg.get(eid, 0)
            if already + d >= e["hp"]:
                best = max(best, 20.0)
            else:
                etile = terrain[e["y"]][e["x"]]
                pressure = etile * 1.0 + (3.0 if (e["x"], e["y"]) in obj else 0.0)
                best = max(best, 1.0 + pressure * 0.5)
        return best

    def score_tile(t):
        return tile_value(t) + combat_bonus(t)

    best_t = start
    best_s = score_tile(start)
    for t in reach:
        if t == start:
            continue
        s = score_tile(t)
        if s > best_s + 1e-9:
            best_s = s
            best_t = t
    dest = best_t
    occupied_f.discard(start)
    occupied_f.add(dest)
    u["x"], u["y"] = dest

    adj = [eid for eid, e in enemy.items() if e["hp"] > 0 and abs(e["x"] - dest[0]) + abs(e["y"] - dest[1]) == 1]
    atk = None
    if adj:
        def kill_first(eid):
            e = enemy[eid]
            d = dmg(u["atk"], terrain[e["y"]][e["x"]])
            already = pending_dmg.get(eid, 0)
            finishes = 1 if already + d >= e["hp"] else 0
            pressure = terrain[e["y"]][e["x"]] + (3 if (e["x"], e["y"]) in obj else 0)
            return (-finishes, -pressure, e["hp"], eid)
        atk = min(adj, key=kill_first)
        e = enemy[atk]
        d = dmg(u["atk"], terrain[e["y"]][e["x"]])
        pending_dmg[atk] = pending_dmg.get(atk, 0) + d
    orders_out.append({"unit_id": fid, "move_to": (None if dest == start else list(dest)), "attack": atk})

print(json.dumps({"orders": orders_out, "state": None}))
