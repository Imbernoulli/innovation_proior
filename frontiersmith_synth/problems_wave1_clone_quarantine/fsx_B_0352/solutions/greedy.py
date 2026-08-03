# TIER: greedy
# Frequency-first layout: place the predicates that appear in the MOST checks at the
# front of the global order (ties broken by heavier token weight). Frequent core
# predicates thus lead every check's prefix while the rare sprinkles fall to the back,
# so same-family checks share a long common prefix. A strong, natural heuristic.
import sys, json
inst = json.load(sys.stdin)
M = inst["M"]
w = inst["weights"]
Q = inst["queries"]
freq = [0] * M
for q in Q:
    for a in q:
        freq[a] += 1
order = sorted(range(M), key=lambda a: (-freq[a], -w[a], a))
print(json.dumps({"order": order}))
