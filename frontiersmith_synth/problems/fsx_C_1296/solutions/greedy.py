# TIER: greedy
# The obvious first instinct for "explore and collect as much as possible":
# breadth-first the room graph (ties broken by ascending room id) and, the
# instant you step into a room, interact with EVERY fixture there right away,
# before moving on -- explore fast, grab everything, never pause to read a
# field-note cue or worry about what a fixture might need first. This clears
# every trap-free warm-up cleanly, but the moment it steps on an irreversible
# fixture with no ward in hand (which happens almost immediately in most
# worlds -- the dangerous fixture usually sits right next to the start), the
# expedition ends on the spot and the rest of the plan is never executed.
import sys, json

inst = json.load(sys.stdin)
R = inst["rooms"]
start = inst["start"]

adj = {i: [] for i in range(R)}
for a, b in inst["edges"]:
    adj[a].append(b)
    adj[b].append(a)
for i in range(R):
    adj[i].sort()

by_room = {}
for o in inst["objects"]:
    by_room.setdefault(o["room"], []).append(o["id"])
for r in by_room:
    by_room[r].sort()

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

print(json.dumps({"actions": actions}))
