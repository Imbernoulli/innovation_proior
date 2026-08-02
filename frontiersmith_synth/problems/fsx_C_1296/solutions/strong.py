# TIER: strong
# The insight: a fixture's field-note "cue" always carries exactly one
# adjective from one of two fixed pools (documented in the statement) --
# DECAY words mean the fixture is irreversible (safe only with a ward in
# hand), STABLE words mean it is reversible (always safe). You never need to
# gamble to find out; you can read it off the grammar before committing.
#
# Two-phase plan, an exchange-argument allocation instead of "greedy + more
# iterations":
#   1. SAFE-CLOSURE PASS: BFS the whole map and interact with every fixture
#      classified SAFE (and only those) -- this can never fail, and it is
#      exactly where the wards live, so by the end of this pass you know your
#      true ward budget.
#   2. WARD ALLOCATION: among the fixtures classified RISKY, sort by reward
#      descending and spend wards on the top ones first. Because every risky
#      fixture costs exactly one ward regardless of its reward, sorting by
#      value and taking a prefix is provably optimal for a fixed ward supply
#      (a textbook exchange argument: swapping a lower-reward pick for a
#      higher-reward one already-not-picked never decreases the total) --
#      the genuine allocation insight, not just "avoid what looks risky".
import sys, json

DECAY_WORDS = ["crumbling", "trembling", "creaking", "hollow-sounding",
               "frayed", "splintered", "rusted-through", "warped"]
STABLE_WORDS = ["sturdy", "gleaming", "freshly-oiled", "solid",
                "well-anchored", "polished", "intact", "firmly-set"]


def classify(cue):
    for w in DECAY_WORDS:
        if w in cue:
            return True   # risky / irreversible
    for w in STABLE_WORDS:
        if w in cue:
            return False  # safe / reversible
    return True  # unrecognized grammar: treat cautiously as risky


inst = json.load(sys.stdin)
R = inst["rooms"]
start = inst["start"]

adj = {i: [] for i in range(R)}
for a, b in inst["edges"]:
    adj[a].append(b)
    adj[b].append(a)
for i in range(R):
    adj[i].sort()

objects = inst["objects"]
by_room = {}
for o in objects:
    by_room.setdefault(o["room"], []).append(o)
risky_of = {o["id"]: classify(o["cue"]) for o in objects}

# BFS visiting order of every room (used for the safe-closure sweep: we walk
# the whole map since moving is always free of risk, only fixtures are not).
seen = {start}
q = [start]
head = 0
while head < len(q):
    u = q[head]; head += 1
    for v in adj[u]:
        if v not in seen:
            seen.add(v)
            q.append(v)
order = q[1:]

actions = []
used = set()

# Phase 1: visit every room, interact only with fixtures classified SAFE.
for r in order:
    actions.append({"type": "goto", "room": r})
    for o in by_room.get(r, []):
        if not risky_of[o["id"]]:
            actions.append({"type": "interact", "object": o["id"]})
            used.add(o["id"])
for o in by_room.get(start, []):
    if not risky_of[o["id"]] and o["id"] not in used:
        actions.insert(0, {"type": "interact", "object": o["id"]})
        used.add(o["id"])

# Phase 2: spend wards on the highest-reward RISKY fixtures first.
risky_objs = [o for o in objects if risky_of[o["id"]] and o["id"] not in used]
risky_objs.sort(key=lambda o: -o["reward"])
for o in risky_objs:
    actions.append({"type": "goto", "room": o["room"]})
    actions.append({"type": "interact", "object": o["id"]})

print(json.dumps({"actions": actions}))
