# TIER: greedy
# Single-signal heuristic: sort candidates by the universal quality prior, the
# star-rating feature (index 0), highest first.  Rating has a positive taste weight
# in every session, so this reliably beats the as-presented order -- but it ignores
# the session-specific tastes (price / popularity / discount / freshness / affinity),
# so it leaves gains on the table versus a learned model.
import sys, json

inst = json.load(sys.stdin)
items = inst["items"]
m = len(items)

order = sorted(range(m), key=lambda i: (items[i][0], i), reverse=True)
print(json.dumps({"ranking": order}))
