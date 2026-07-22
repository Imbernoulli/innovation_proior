# TIER: trivial
# Do the absolute minimum: buy exactly one batch (one lot) of raw type 0 and nothing
# else, completely ignoring the target ratio, the other raw types, and every
# mechanism. A valid but weak answer.
import sys, json

inst = json.load(sys.stdin)
P = inst["P"]
order = [0] * P
order[0] = inst["lot"][0]
print(json.dumps({"order": order}))
