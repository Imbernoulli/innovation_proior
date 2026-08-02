#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1296 -- "Mapping a World Through a Keyhole"
(family: text-adventure-explore; format B, quality-metric).

THEME.  A cartographer's expedition planner is handed a partially-surveyed dungeon
map: the ROOM TOPOLOGY is fully drawn (rooms + passages), but the true nature of
every interactive fixture (a bridge, a chest, a lever, ...) is only rumored --
field-notes ("cues") hint at whether it is safe or dangerous, but never say so
outright.  Some fixtures are REVERSIBLE to probe (safe: they always pay off).
Others are IRREVERSIBLE risks: attempting one without a protective charm ("ward")
in hand ends the expedition on the spot (everything planned afterward is lost).
Wards are themselves scarce, found only by opening safe fixtures elsewhere, so the
planner must sequence a route that gathers wards BEFORE spending them on the
fixtures worth the gamble -- an inventory-precondition-graph on top of the risk.

The planner never gets to watch the expedition unfold turn by turn (partial
observability + a single committed plan): it must commit to a full itinerary in
advance, using only the map, the fixtures' field-note cues, and the documented
cue grammar below -- NOT the true safe/risky flag, which stays hidden in the
judge and is only used to score the submitted itinerary.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance) -- see statement.md for the schema
          (rooms, edges, turn_budget, objects[] with {id, room, verb, cue, reward,
          gives_item} -- note NO "risky" field: that is hidden).
  stdout: ONE JSON object: {"actions": [ {"type":"goto","room":R}, ... |
                                          {"type":"interact","object":ID}, ... ]}

CUE GRAMMAR (documented, honest, identical across every instance):
  A fixture's cue always contains exactly one adjective drawn from one of two
  fixed pools:
    DECAY_WORDS (the fixture is IRREVERSIBLE / risky) = {crumbling, trembling,
      creaking, hollow-sounding, frayed, splintered, rusted-through, warped}
    STABLE_WORDS (the fixture is REVERSIBLE / safe)    = {sturdy, gleaming,
      freshly-oiled, solid, well-anchored, polished, intact, firmly-set}
  This mapping is 100% reliable and IDENTICAL for every instance in this family:
  it is the mechanism, not a tunable secret.  Nothing else in the public JSON
  reveals the true type (gives_item is present on many safe AND many "would be
  risky" flavor objects and is NOT a reliable tell by itself).

SIMULATION (deterministic; this parent process, never the candidate, holds the
hidden risky/safe flags).  Starting at the instance's start room with 0 turns
used, 0 wards held, the judge executes the submitted actions in order:
  - {"type":"goto","room":R}: costs `shortest_path_hops(current, R)` turns
    (computed on the PUBLIC, fully-known room graph); moves there if the turn
    budget allows, else the step is a 1-turn no-op.
  - {"type":"interact","object":ID}: costs 1 turn.  No-op (but still costs the
    turn) if the object isn't in the current room or was already used.
    Otherwise: if the fixture is SAFE, it always succeeds (reward added, and if
    it grants a ward, ward count += 1).  If the fixture is RISKY: succeeds
    (reward added, ward consumed) only if a ward is currently held; otherwise
    the expedition ends immediately -- the rest of the submitted itinerary is
    never executed.
  Execution also stops once the turn budget is exhausted.
  Objective per instance = (# distinct rooms visited) * ROOM_BONUS + (total
  reward collected).

SCORING (deterministic; no wall-time).  Per instance the judge computes, itself:
    q_base = objective of the "exhaustive blind" reference itinerary: BFS the
             room graph (ties broken by ascending room id) and, the instant a
             room is first entered, interact with every fixture in it (in id
             order) before continuing -- i.e. explore everything, grab
             everything, never pause to read a cue.  This is what an "explore
             fast, try everything" first instinct produces; on trapped
             instances it detonates the very first risky fixture it meets.
    q_ub   = an optimistic, generally-unreachable upper bound: every room
             visited, every safe reward collected, plus the top
             min(#wards-in-the-world, #risky) risky rewards by value -- with
             NO travel cost and NO turn budget.  (Loose, like an L1 bin-packing
             bound: real itineraries pay for travel and are budget-limited, so
             even a perfect planner stays under this.)
    q_cand = objective of the candidate's submitted itinerary.
  Normalized with an affine anchor (blind-exhaustive baseline -> 0.1, the
  optimistic bound -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(1e-9, q_ub - q_base), 0, 1 )

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance (no "risky" flags,
no q_base/q_ub).  All references and validation happen in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun

ROOM_BONUS = 2.0

NOUNS = [
    "stone bridge", "iron-bound chest", "brass lever", "carved door",
    "rope ladder", "stone idol", "old well", "dusty shrine", "rusty hatch",
    "garden gate", "wooden crate", "glass cabinet", "ivy-choked archway",
    "copper valve",
]
DECAY_WORDS = ["crumbling", "trembling", "creaking", "hollow-sounding",
               "frayed", "splintered", "rusted-through", "warped"]
STABLE_WORDS = ["sturdy", "gleaming", "freshly-oiled", "solid",
                "well-anchored", "polished", "intact", "firmly-set"]
VERBS = {"stone bridge": "cross", "iron-bound chest": "open", "brass lever": "pull",
         "carved door": "open", "rope ladder": "climb", "stone idol": "touch",
         "old well": "descend", "dusty shrine": "pray at", "rusty hatch": "open",
         "garden gate": "push", "wooden crate": "pry open", "glass cabinet": "open",
         "ivy-choked archway": "step through", "copper valve": "turn"}


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


def _cue(rng, risky):
    noun = NOUNS[rng(0, len(NOUNS) - 1)]
    pool = DECAY_WORDS if risky else STABLE_WORDS
    adj = pool[rng(0, len(pool) - 1)]
    return "Through the keyhole you note: the %s looks %s." % (noun, adj), VERBS[noun]


# ----------------------------- world specs ----------------------------------
# Each world: (name, seed, R, edges, start, budget, objects)
# objects: list of (room, risky, reward, gives_item) ; id = index in this list.
_WORLDS = [
    # ---- 3 trap-free warm-ups: no risky fixtures at all ----
    ("atrium", 101, 6,
     [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5)], 0, 24,
     [(1, False, 6, None), (2, False, 5, None), (3, False, 7, None),
      (4, False, 4, None), (5, False, 6, None)]),
    ("cloister", 102, 7,
     [(0, 1), (1, 2), (1, 3), (0, 4), (4, 5), (5, 6)], 0, 28,
     [(2, False, 5, None), (3, False, 6, None), (4, False, 4, None),
      (5, False, 5, None), (6, False, 8, None)]),
    ("grotto", 103, 5,
     [(0, 1), (0, 2), (2, 3), (2, 4)], 0, 16,
     [(1, False, 7, None), (3, False, 5, None), (4, False, 6, None)]),

    # ---- 7 trapped worlds: a risky fixture sits in start's BFS layer-1
    #      neighborhood, before any ward source is ever reached blind ----
    ("collapsed-nave", 204, 8,
     [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (6, 7)], 0, 32,
     [(1, True, 9, None), (2, False, 5, None), (3, False, 6, None),
      (4, False, 4, "ward"), (5, False, 5, None), (6, True, 8, None),
      (7, False, 7, None)]),
    ("sunken-crypt", 205, 9,
     [(0, 1), (1, 2), (1, 3), (0, 4), (4, 5), (4, 6), (6, 7), (7, 8)], 0, 24,
     [(1, True, 10, None), (2, False, 5, None), (3, False, 6, None),
      (4, False, 4, "ward"), (5, False, 5, None), (6, True, 7, None),
      (7, False, 6, "ward"), (8, False, 8, None)]),
    ("vault-of-echoes", 206, 10,
     [(0, 1), (0, 2), (2, 3), (2, 4), (4, 5), (4, 6), (6, 7), (6, 8), (8, 9)], 0, 20,
     [(1, False, 5, None), (2, True, 9, None), (3, False, 5, None),
      (4, False, 4, "ward"), (5, False, 6, None), (6, True, 8, None),
      (7, False, 5, "ward"), (8, False, 7, None), (9, False, 9, None)]),
    ("flooded-archive", 207, 8,
     [(0, 1), (1, 2), (0, 3), (3, 4), (4, 5), (0, 6), (6, 7)], 0, 20,
     [(1, True, 3, None), (2, False, 5, None), (3, False, 6, "ward"),
      (4, True, 9, None), (5, False, 5, None), (6, True, 4, None),
      (7, False, 6, None)]),
    ("far-archive", 308, 12,
     [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (6, 7),
      (0, 8), (8, 9), (9, 10), (10, 11)], 0, 30,
     [(1, True, 9, None), (2, False, 5, None), (3, False, 6, None),
      (4, False, 5, "ward"), (5, True, 10, None), (6, False, 5, None),
      (7, False, 6, None), (8, False, 4, None), (9, True, 7, None),
      (10, False, 6, "ward"), (11, False, 8, None)]),
    ("spiral-stacks", 309, 13,
     [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7),
      (0, 8), (8, 9), (9, 10), (10, 11), (11, 12)], 0, 34,
     [(1, True, 8, None), (2, False, 5, None), (3, False, 5, "ward"),
      (4, False, 6, None), (5, False, 4, None), (6, True, 10, None),
      (7, False, 5, "ward"), (8, False, 4, None), (9, True, 6, None),
      (10, False, 5, None), (11, False, 6, None), (12, False, 7, None)]),
    ("the-last-keyhole", 410, 14,
     [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
      (0, 9), (9, 10), (10, 11), (11, 12), (12, 13)], 0, 30,
     [(1, True, 5, None), (2, False, 5, "ward"), (3, True, 9, None),
      (4, False, 6, None), (5, False, 4, None), (6, True, 7, None),
      (7, False, 5, "ward"), (8, False, 6, None), (9, False, 4, None),
      (10, True, 11, None), (11, False, 5, None), (12, True, 6, None),
      (13, False, 6, "ward")]),
]


def _build_instances():
    out = []
    for name, seed, R, edges, start, budget, objs in _WORLDS:
        rng = _rng(seed)
        adj = {i: [] for i in range(R)}
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        for i in range(R):
            adj[i].sort()
        objects = []
        for oid, (room, risky, reward, gives) in enumerate(objs):
            cue, verb = _cue(rng, risky)
            objects.append({"id": "o%d" % oid, "room": room, "risky": risky,
                            "reward": reward, "gives_item": gives,
                            "cue": cue, "verb": verb})
        out.append({"name": name, "R": R, "adj": adj, "start": start,
                    "budget": budget, "objects": objects})
    return out


# ----------------------------- shortest paths -------------------------------
def _bfs_dist(adj, src):
    dist = {src: 0}
    q = [src]
    head = 0
    while head < len(q):
        u = q[head]; head += 1
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


# ----------------------------- simulation ------------------------------------
def _simulate(inst, actions):
    adj = inst["adj"]
    R = inst["R"]
    by_room = {}
    by_id = {}
    for o in inst["objects"]:
        by_room.setdefault(o["room"], []).append(o)
        by_id[o["id"]] = o
    pos = inst["start"]
    budget = inst["budget"]
    turns = 0
    visited = {pos}
    reward = 0
    wards = 0
    used = set()
    alive = True
    dist_cache = {}
    for act in actions:
        if turns >= budget or not alive:
            break
        if not isinstance(act, dict):
            continue
        typ = act.get("type")
        if typ == "goto":
            tgt = act.get("room")
            if isinstance(tgt, bool) or not isinstance(tgt, int) or tgt < 0 or tgt >= R:
                turns += 1
                continue
            if tgt == pos:
                continue
            if pos not in dist_cache:
                dist_cache[pos] = _bfs_dist(adj, pos)
            d = dist_cache[pos].get(tgt)
            if d is None:
                turns += 1
                continue
            if turns + d > budget:
                turns += 1
                continue
            turns += d
            pos = tgt
            visited.add(pos)
        elif typ == "interact":
            oid = act.get("object")
            turns += 1
            if not isinstance(oid, str):
                continue
            obj = by_id.get(oid)
            if obj is None or obj["room"] != pos or oid in used:
                continue
            used.add(oid)
            if obj["risky"]:
                if wards > 0:
                    wards -= 1
                    reward += obj["reward"]
                    if obj["gives_item"] == "ward":
                        wards += 1
                else:
                    alive = False
            else:
                reward += obj["reward"]
                if obj["gives_item"] == "ward":
                    wards += 1
        else:
            turns += 1
    return len(visited) * ROOM_BONUS + reward


def _greedy_plan(inst):
    """Exhaustive-blind reference: BFS the graph (ascending room id), and the
    instant a room is entered, interact with every fixture there (id order)
    before moving on. Never reads a cue."""
    adj = inst["adj"]
    start = inst["start"]
    by_room = {}
    for o in inst["objects"]:
        by_room.setdefault(o["room"], []).append(o["id"])
    for r in by_room:
        by_room[r].sort()
    order = []
    seen = {start}
    q = [start]
    head = 0
    while head < len(q):
        u = q[head]; head += 1
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    order = q[1:]  # BFS order excluding start
    actions = []
    for r in order:
        actions.append({"type": "goto", "room": r})
        for oid in by_room.get(r, []):
            actions.append({"type": "interact", "object": oid})
    for oid in by_room.get(start, []):
        actions.insert(0, {"type": "interact", "object": oid})
    return actions


def _ub(inst):
    safe_sum = sum(o["reward"] for o in inst["objects"] if not o["risky"])
    n_wards = sum(1 for o in inst["objects"] if (not o["risky"]) and o["gives_item"] == "ward")
    risky_rewards = sorted((o["reward"] for o in inst["objects"] if o["risky"]), reverse=True)
    risky_sum = sum(risky_rewards[:n_wards])
    return inst["R"] * ROOM_BONUS + safe_sum + risky_sum


# ----------------------------- validation ------------------------------------
def _score_answer(inst, answer):
    if not isinstance(answer, dict):
        return None
    actions = answer.get("actions")
    if not isinstance(actions, list) or len(actions) > 500:
        return None
    for a in actions:
        if not isinstance(a, dict):
            return None
    return _simulate(inst, actions)


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        q_base = _simulate(inst, _greedy_plan(inst))
        q_ub = _ub(inst)
        denom = q_ub - q_base
        if denom < 1e-9:
            denom = 1e-9
        public = {
            "name": inst["name"], "rooms": inst["R"], "start": inst["start"],
            "turn_budget": inst["budget"],
            "edges": [[a, b] for a, bs in inst["adj"].items() for b in bs if a < b],
            "objects": [
                {"id": o["id"], "room": o["room"], "verb": o["verb"],
                 "cue": o["cue"], "reward": o["reward"], "gives_item": o["gives_item"]}
                for o in inst["objects"]
            ],
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            q_cand = _score_answer(inst, ans)
        except Exception:
            q_cand = None
        if q_cand is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (q_cand - q_base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
