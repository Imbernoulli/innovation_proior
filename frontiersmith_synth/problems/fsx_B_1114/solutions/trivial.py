# TIER: trivial
# Memorize the first faces you ever see, fill the list once, then freeze it
# forever. Never re-learn, never forget. Reproduces the evaluator's own
# calibrated baseline() exactly.
import sys, json

inst = json.load(sys.stdin)
capacity = inst["capacity"]
floor = set(inst["floor"])
arrivals = inst["arrivals"]

decisions = []
for key in arrivals:
    if key in floor:
        decisions.append({"action": "skip", "evict": None})
        continue
    if len(floor) < capacity:
        floor.add(key)
        decisions.append({"action": "admit", "evict": None})
    else:
        decisions.append({"action": "skip", "evict": None})

print(json.dumps({"decisions": decisions, "state": None}))
