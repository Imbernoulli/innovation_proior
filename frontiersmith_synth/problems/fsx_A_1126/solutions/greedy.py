# TIER: greedy
# The "obvious" recipe: a clerk who has learned to prioritise her most
# valuable guests. Sort guests by value DESCENDING (ties by id ascending for
# determinism), first-fit each into the first room whose current occupants
# have no clash with it (checked against the public graph), else turn the
# guest away. This clearly beats seating in raw roster order (it reuses gaps
# and protects high-value guests individually) -- but it reasons about VALUE
# ALONE, never about how much room-time a guest consumes. A single
# high-value guest whose stay spans an entire room for its whole duration
# gets seated first and can block many smaller, collectively far more
# valuable, guests for the rest of the run: it never reconsiders.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
R = inst["rooms"]
edges = inst["edges"]
value = inst["value"]

adj = [set() for _ in range(n)]
for a, b in edges:
    adj[a].add(b)
    adj[b].add(a)

order = sorted(range(n), key=lambda i: (-value[i], i))

room_members = [set() for _ in range(R)]
room = [-1] * n
for i in order:
    for r in range(R):
        if not (adj[i] & room_members[r]):
            room_members[r].add(i)
            room[i] = r
            break

print(json.dumps({"room": room}))
