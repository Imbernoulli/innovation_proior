# TIER: greedy
# Plain global LRU: admit everyone, evict the least-recently-used resident
# when full. The obvious first move for a strong coder. Reacts to raw
# recency only -- it never reads scan_period/scan_span/inversion_period, so
# a single tour-bus round evicts the entire memorized list (a bus round's
# one-shot faces are, by definition, the most recent thing seen), and it
# cannot tell a "regular due back next rotation" apart from a face that is
# gone for good, so it lets the wrong half of the list get evicted around
# every crowd rotation.
import sys, json

inst = json.load(sys.stdin)
capacity = inst["capacity"]
floor = set(inst["floor"])
arrivals = inst["arrivals"]
state = inst.get("state") or {}
clock = int(state.get("clock", 0))
recency = {k: v for k, v in (state.get("recency") or {}).items() if k in floor}

decisions = []
for key in arrivals:
    clock += 1
    if key in floor:
        recency[key] = clock
        decisions.append({"action": "skip", "evict": None})
        continue
    if len(floor) < capacity:
        floor.add(key)
        recency[key] = clock
        decisions.append({"action": "admit", "evict": None})
    else:
        victim = min(floor, key=lambda k: recency.get(k, -1))
        floor.discard(victim)
        recency.pop(victim, None)
        floor.add(key)
        recency[key] = clock
        decisions.append({"action": "admit", "evict": victim})

print(json.dumps({"decisions": decisions, "state": {"clock": clock, "recency": recency}}))
