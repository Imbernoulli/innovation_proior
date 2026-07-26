# TIER: trivial
# WEAK reference clerk: seat guests in the order the (already scrambled)
# roster lists them, first-fit into rooms 0..R-1 using only the public clash
# graph. Never reorders by value, never looks at structure. Reproduces the
# evaluator's own "base" reference, so it scores ~0.1.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
R = inst["rooms"]
edges = inst["edges"]

adj = [set() for _ in range(n)]
for a, b in edges:
    adj[a].add(b)
    adj[b].add(a)

room_members = [set() for _ in range(R)]
room = [-1] * n
for i in range(n):
    for r in range(R):
        if not (adj[i] & room_members[r]):
            room_members[r].add(i)
            room[i] = r
            break

print(json.dumps({"room": room}))
