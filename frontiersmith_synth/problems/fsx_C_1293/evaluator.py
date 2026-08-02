import sys, json, isorun
from collections import deque

# ==========================================================================
# fsx_C_1293 -- tactical-squad-policy (Format B, isolated candidate)
# Theme: "Units that win by not fighting fair"
#
# Mechanisms composed into ONE objective:
#   (1) terrain-defense-bonus  -- every tile mitigates incoming damage by its
#       defense_bonus (0/1/2); the SAME tile also determines how much damage
#       an attacker on it deals when it attacks (occupying a hill is a
#       shield for you and a bigger sword whenever you swing FROM it -- see
#       dmg() using the DEFENDER's tile in both directions symmetrically).
#   (2) focus-fire-vs-spread   -- all attacks ordered THIS turn are pooled
#       per target and applied simultaneously (see dmg_pool). A target that
#       dies this turn contributes NO attack in the enemy phase and can no
#       longer occupy/deny a tile; a target merely wounded gets a full
#       attack back and keeps denying position. Spreading chip damage across
#       many weak targets (kills none) is strictly worse than finishing one.
#   (3) zone-of-control (ZoC)  -- every living unit projects a ZoC onto its 4
#       orthogonal neighbours. A mover may step INTO a ZoC tile (that is
#       always a legal final stop) but may never move FURTHER once inside
#       one -- you cannot dash past/around a guarded doorway; you can only
#       either commit to the fight there or route around it entirely.
#
# INNOVATION HOOK. Two designated "objective" tiles (always hills, i.e.
# defense_bonus==2) are worth HOLD_REWARD points every turn-end a living
# friendly unit stands on them. The obvious first move -- "each turn, chase
# and attack whichever ENEMY currently has the globally lowest HP, ignoring
# what tile that requires standing on" (solutions/greedy.py) -- maximizes
# immediate kills but systematically abandons the hills the instant *any*
# enemy becomes reachable, and it fights on whatever open ground the chase
# ends on. The winning policy (solutions/strong.py) instead scores every
# reachable tile by the defense/objective value it would grab -- and, just
# as importantly, DENY the enemy for the next several turns -- only
# preferring an attack over holding ground when that attack finishes a kill
# outright (pending_dmg pooling) or the ground gained is not actually worth
# more than what is already held. Position value compounds over the
# remaining turns; a kill's value is a one-turn event.
#
# PROTOCOL. One "battle" = one instance = a fixed sequence of TURNS on a
# small grid. The candidate program is invoked ONCE PER TURN, isolated, no
# shared process memory between calls -- any bookkeeping it wants (recency,
# plans, whatever) must ride in the opaque "state" field it emits and gets
# back verbatim next turn. Each call's stdin is the PUBLIC view for that
# turn only: full terrain grid, the two objective-tile coordinates, every
# CURRENTLY LIVING friendly unit (with hp/atk/move) and every currently
# living enemy unit (with hp/atk/move -- the fixed enemy AI's own stats are
# handed over verbatim, nothing to infer), and the candidate's own last
# "state". The candidate returns one order per living friendly unit
# (move_to a reachable tile or null-to-stay, plus an optional attack target)
# and its next "state".
#
# ENEMY AI (fixed, deterministic, described here in full -- nothing hidden):
# each surviving enemy, in ascending id order, if it is not already adjacent
# to some living friendly unit, moves (respecting the same ZoC rule) to the
# reachable tile that minimizes Manhattan distance to the CURRENT nearest
# living friendly unit (ties: lower resulting distance, then smaller y then
# smaller x). After all enemy moves, every enemy adjacent to at least one
# living friendly unit contributes damage -- simultaneously pooled exactly
# like the friendly phase -- to the reachable friendly unit with the lowest
# current HP (ties: lower id).
#
# SCORING. Per battle: raw = (turns-held * objective tiles held, summed) +
# 0.4*(surviving friendly HP / 10) + 1.35*(enemies killed). A trivial policy
# that never moves and never attacks (solutions/trivial.py) is always valid
# and is exactly what the grader's own baseline() computes -- it reproduces
# the calibrated 0.1 ratio exactly.
# ==========================================================================

STATE_MAX_CHARS = 40000
HOLD_REWARD = 1.0
PRESENCE_WEIGHT = 0.4
KILL_WEIGHT = 1.35


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
    """BFS over the grid: `start` is always reachable. A tile OTHER than
    `start` that lies in an opposing unit's zone of control is a legal
    final stop but cannot be expanded further (you cannot move past it)."""
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


def mk_map(name, W, H, walls, hills, forest, friendly, enemy, T=8):
    terrain = [[0] * W for _ in range(H)]
    for (x, y) in walls:
        terrain[y][x] = -1
    for (x, y) in forest:
        terrain[y][x] = 1
    for (x, y) in hills:
        terrain[y][x] = 2
    fr = [{"id": f"f{i}", "x": u[0], "y": u[1], "hp": u[2], "atk": u[3], "move": u[4]}
          for i, u in enumerate(friendly)]
    en = [{"id": f"e{i}", "x": u[0], "y": u[1], "hp": u[2], "atk": u[3], "move": u[4]}
          for i, u in enumerate(enemy)]
    return {"name": name, "W": W, "H": H, "terrain": terrain,
            "objective_tiles": [list(h) for h in hills],
            "friendly0": fr, "enemy0": en, "T": T}


def unit(x, y, hp=10, atk=4, mv=3):
    return (x, y, hp, atk, mv)


# ----------------------------- instance generator ---------------------------
def make_instances():
    """10 fixed, hand-designed battles (no randomness needed -- the maps
    themselves are the deterministic seed). Instances 0-2 are open-terrain
    warm-ups (small defense differential, both squads meet in a compact
    line so a damage-chaser is not badly punished for chasing). Instances
    3-9 are terrain-rich: single-tile chokepoints that turn zone-of-control
    into a real constraint, and/or hills whose defense bonus swings combat
    trades hard enough that abandoning them to chase a weak kill costs far
    more than the kill is worth."""
    specs = []

    specs.append(mk_map(
        "open_a", 10, 6, walls=[], hills=[(3, 2), (3, 3)], forest=[],
        friendly=[unit(3, 2), unit(3, 3), unit(4, 2), unit(4, 3)],
        enemy=[unit(6, 2), unit(6, 3), unit(7, 2), unit(7, 3)],
        T=8))

    specs.append(mk_map(
        "open_b", 10, 6, walls=[], hills=[(3, 2), (3, 3)], forest=[],
        friendly=[unit(3, 2), unit(3, 3), unit(4, 2), unit(4, 3)],
        enemy=[unit(6, 1), unit(6, 2), unit(6, 3), unit(6, 4), unit(7, 2)],
        T=8))

    specs.append(mk_map(
        "open_c", 11, 6, walls=[], hills=[(4, 2), (4, 3)], forest=[],
        friendly=[unit(4, 2), unit(4, 3), unit(5, 2), unit(5, 3), unit(5, 4)],
        enemy=[unit(7, 1), unit(7, 2), unit(7, 3), unit(7, 4)],
        T=9))

    specs.append(mk_map(
        "trap_corridor_a", 9, 7,
        walls=[(4, 0), (4, 1), (4, 2), (4, 4), (4, 5), (4, 6)],
        hills=[(2, 2), (2, 4)], forest=[],
        friendly=[unit(2, 2), unit(2, 4), unit(1, 3), unit(0, 3)],
        enemy=[unit(6, 3), unit(7, 1), unit(7, 5), unit(8, 3)],
        T=9))

    specs.append(mk_map(
        "trap_corridor_b", 9, 7,
        walls=[(4, 0), (4, 1), (4, 2), (4, 4), (4, 5), (4, 6)],
        hills=[(2, 3), (1, 1)], forest=[(1, 5)],
        friendly=[unit(2, 3), unit(1, 1), unit(0, 3), unit(1, 4)],
        enemy=[unit(5, 3), unit(7, 0), unit(7, 2), unit(7, 4), unit(7, 6)],
        T=9))

    specs.append(mk_map(
        "trap_open_hill_a", 9, 6, walls=[], hills=[(4, 2), (4, 3)], forest=[(3, 1), (5, 4)],
        friendly=[unit(4, 2), unit(4, 3), unit(3, 3), unit(5, 2)],
        enemy=[unit(6, 1, hp=6), unit(6, 4, hp=6), unit(7, 2), unit(7, 3)],
        T=9))

    specs.append(mk_map(
        "trap_open_hill_b", 10, 6, walls=[], hills=[(5, 2), (5, 3)], forest=[(4, 1), (6, 4)],
        friendly=[unit(5, 2), unit(5, 3), unit(4, 3), unit(6, 2), unit(4, 2)],
        enemy=[unit(7, 1, hp=6), unit(7, 4, hp=6), unit(8, 2), unit(8, 3), unit(8, 5)],
        T=9))

    specs.append(mk_map(
        "trap_double_door", 10, 8,
        walls=[(4, 0), (4, 1), (4, 3), (4, 4), (4, 6), (4, 7)],
        hills=[(2, 2), (2, 5)], forest=[(1, 3), (1, 4)],
        friendly=[unit(2, 2), unit(2, 5), unit(1, 2), unit(1, 5)],
        enemy=[unit(6, 2), unit(6, 5), unit(8, 0), unit(8, 7), unit(9, 3)],
        T=10))

    specs.append(mk_map(
        "trap_skirmish_bait", 10, 6, walls=[], hills=[(3, 2), (3, 3)], forest=[],
        friendly=[unit(3, 2), unit(3, 3), unit(4, 2), unit(4, 3)],
        enemy=[unit(6, 0, hp=5), unit(6, 5, hp=5), unit(7, 2), unit(7, 3)],
        T=9))

    specs.append(mk_map(
        "held_out_large", 11, 8,
        walls=[(5, 0), (5, 1), (5, 2), (5, 5), (5, 6), (5, 7)],
        hills=[(2, 3), (2, 4)], forest=[(1, 1), (1, 6)],
        friendly=[unit(2, 3), unit(2, 4), unit(1, 3), unit(1, 4), unit(0, 3), unit(0, 4)],
        enemy=[unit(6, 3), unit(6, 4), unit(8, 0), unit(8, 7), unit(9, 3)],
        T=11))

    return [{"public": None, "hidden": h} for h in specs]


# ----------------------------- battle simulation -----------------------------
def simulate(hidden, order_fn):
    """Runs a full multi-turn battle against order_fn(public)->answer|None.
    Returns (ok, raw_score). Any structural violation at the whole-answer
    level (not a dict / not a list of the right length / duplicate or
    missing unit ids / oversized state) invalidates the ENTIRE battle -> 0.
    A well-formed but tactically infeasible single order (illegal
    destination, out-of-range attack target) is simply ignored for that one
    unit (it stays put / does not attack) rather than voiding the battle."""
    W, H = hidden["W"], hidden["H"]
    terrain = hidden["terrain"]
    obj_tiles = set(tuple(t) for t in hidden["objective_tiles"])
    friendly = {u["id"]: dict(u) for u in hidden["friendly0"]}
    enemy = {u["id"]: dict(u) for u in hidden["enemy0"]}
    friendly_order = [u["id"] for u in hidden["friendly0"]]
    enemy_order = [u["id"] for u in hidden["enemy0"]]
    T = hidden["T"]
    state = None
    hold_points = 0.0

    for turn in range(T):
        alive_f = [fid for fid in friendly_order if friendly[fid]["hp"] > 0]
        alive_e = [eid for eid in enemy_order if enemy[eid]["hp"] > 0]
        if not alive_f:
            break
        public = {
            "turn": turn, "total_turns": T, "W": W, "H": H,
            "terrain": terrain, "objective_tiles": [list(t) for t in obj_tiles],
            "friendly": [{"id": fid, "x": friendly[fid]["x"], "y": friendly[fid]["y"],
                          "hp": friendly[fid]["hp"], "atk": friendly[fid]["atk"], "move": friendly[fid]["move"]}
                         for fid in alive_f],
            "enemy": [{"id": eid, "x": enemy[eid]["x"], "y": enemy[eid]["y"],
                       "hp": enemy[eid]["hp"], "atk": enemy[eid]["atk"], "move": enemy[eid]["move"]}
                      for eid in alive_e],
            "state": state,
        }
        ans = order_fn(public)
        if not isinstance(ans, dict):
            return False, 0.0
        orders = ans.get("orders")
        st = ans.get("state", None)
        try:
            if st is not None and len(json.dumps(st)) > STATE_MAX_CHARS:
                return False, 0.0
        except (TypeError, ValueError):
            return False, 0.0
        if not isinstance(orders, list) or len(orders) != len(alive_f):
            return False, 0.0
        order_by_id = {}
        seen = set()
        ok_struct = True
        for o in orders:
            if not isinstance(o, dict):
                ok_struct = False
                break
            uid = o.get("unit_id")
            if uid not in friendly or uid not in alive_f or uid in seen:
                ok_struct = False
                break
            seen.add(uid)
            order_by_id[uid] = o
        if not ok_struct or seen != set(alive_f):
            return False, 0.0
        state = st

        # ---- friendly movement, sequential in fixed roster order ----
        enemy_positions = set((enemy[eid]["x"], enemy[eid]["y"]) for eid in alive_e)
        for fid in alive_f:
            u = friendly[fid]
            start = (u["x"], u["y"])
            o = order_by_id[fid]
            mv = o.get("move_to", None)
            dest = start
            if isinstance(mv, (list, tuple)) and len(mv) == 2:
                tx, ty = mv
                if (isinstance(tx, int) and isinstance(ty, int)
                        and not isinstance(tx, bool) and not isinstance(ty, bool)):
                    occupied = set((friendly[o2]["x"], friendly[o2]["y"])
                                   for o2 in friendly if friendly[o2]["hp"] > 0 and o2 != fid) | enemy_positions
                    occupied.discard(start)
                    reach = reachable(start, u["move"], occupied, enemy_positions, terrain, W, H)
                    if (tx, ty) in reach:
                        dest = (tx, ty)
            u["x"], u["y"] = dest

        # ---- friendly attacks, pooled per target (focus-fire-vs-spread) ----
        dmg_pool = {}
        alive_e_pos = {eid: (enemy[eid]["x"], enemy[eid]["y"]) for eid in alive_e}
        for fid in alive_f:
            u = friendly[fid]
            o = order_by_id[fid]
            tgt = o.get("attack", None)
            if tgt is None or tgt not in alive_e_pos:
                continue
            ex, ey = alive_e_pos[tgt]
            if abs(u["x"] - ex) + abs(u["y"] - ey) != 1:
                continue
            d = dmg(u["atk"], terrain[ey][ex])
            dmg_pool[tgt] = dmg_pool.get(tgt, 0) + d
        for tgt, dd in dmg_pool.items():
            enemy[tgt]["hp"] -= dd

        alive_e = [eid for eid in enemy_order if enemy[eid]["hp"] > 0]
        alive_f2 = [fid for fid in friendly_order if friendly[fid]["hp"] > 0]

        # ---- fixed enemy AI: move toward nearest living friendly, then
        #      pooled attacks on the lowest-HP reachable friendly ----
        if alive_f2 and alive_e:
            for eid in alive_e:
                e = enemy[eid]
                epos = (e["x"], e["y"])
                alive_f_now = [fid for fid in friendly_order if friendly[fid]["hp"] > 0]
                if not alive_f_now:
                    break

                def dist_to(fid, epos=epos):
                    fpos = (friendly[fid]["x"], friendly[fid]["y"])
                    return abs(fpos[0] - epos[0]) + abs(fpos[1] - epos[1])

                nearest = min(alive_f_now, key=lambda fid: (dist_to(fid), fid))
                if dist_to(nearest) > 1:
                    friendly_pos_set = set((friendly[fid]["x"], friendly[fid]["y"]) for fid in alive_f_now)
                    other_enemy_pos = set((enemy[o2]["x"], enemy[o2]["y"])
                                           for o2 in alive_e if o2 != eid and enemy[o2]["hp"] > 0)
                    occupied = friendly_pos_set | other_enemy_pos
                    reach = reachable(epos, e["move"], occupied, friendly_pos_set, terrain, W, H)
                    npos = (friendly[nearest]["x"], friendly[nearest]["y"])
                    best = min(reach, key=lambda t: (abs(t[0] - npos[0]) + abs(t[1] - npos[1]), t[1], t[0]))
                    e["x"], e["y"] = best

            dmg_pool2 = {}
            friendly_pos_now = {fid: (friendly[fid]["x"], friendly[fid]["y"])
                                 for fid in friendly_order if friendly[fid]["hp"] > 0}
            for eid in alive_e:
                e = enemy[eid]
                candidates = [fid for fid, pos in friendly_pos_now.items()
                              if abs(pos[0] - e["x"]) + abs(pos[1] - e["y"]) == 1]
                if not candidates:
                    continue
                tgt = min(candidates, key=lambda fid: (friendly[fid]["hp"], fid))
                tx, ty = friendly_pos_now[tgt]
                d = dmg(e["atk"], terrain[ty][tx])
                dmg_pool2[tgt] = dmg_pool2.get(tgt, 0) + d
            for tgt, dd in dmg_pool2.items():
                friendly[tgt]["hp"] -= dd

        # ---- end-of-turn objective hold scoring ----
        for fid in friendly_order:
            u = friendly[fid]
            if u["hp"] > 0 and (u["x"], u["y"]) in obj_tiles:
                hold_points += HOLD_REWARD

    lost = sum(1 for fid in friendly_order if friendly[fid]["hp"] <= 0)
    surv_hp = sum(friendly[fid]["hp"] for fid in friendly_order if friendly[fid]["hp"] > 0)
    kills = sum(1 for eid in enemy_order if enemy[eid]["hp"] <= 0)
    raw = hold_points + PRESENCE_WEIGHT * (surv_hp / 10.0) + KILL_WEIGHT * kills
    return True, raw


def trivial_order_fn(public):
    orders = [{"unit_id": u["id"], "move_to": None, "attack": None} for u in public["friendly"]]
    return {"orders": orders, "state": None}


def baseline(inst):
    """Evaluator-computed trivial-construction objective: never move, never
    attack. Pure python, no candidate call -- also exactly what
    solutions/trivial.py implements via the protocol, so a faithful trivial
    candidate reproduces this exactly (ratio == 0.1 on every instance)."""
    ok, raw = simulate(inst["hidden"], trivial_order_fn)
    return raw if ok else 0.0


def make_isorun_order_fn(cand_path):
    def fn(public):
        ans, st = isorun.run_candidate(cand_path, public, timeout=20)
        if st != "OK":
            return None
        return ans
    return fn


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        try:
            ok, obj = simulate(inst["hidden"], make_isorun_order_fn(cand))
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        r = min(1.0, 0.1 * obj / max(b, 1e-9))
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
