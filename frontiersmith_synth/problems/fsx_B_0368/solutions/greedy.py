# TIER: greedy
# Frequency-first layout that IGNORES the season weights: place the predicates that
# appear in the most DISTINCT checks at the front (ties by heavier token weight).
# Frequent core predicates thus lead every check's prefix and rare sprinkles fall to
# the back, so same-family checks share a long common prefix. A strong, natural
# heuristic -- but blind to how often each check actually runs over the season, so it
# under-serves the high-frequency families a season-aware layout would prioritize.
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
