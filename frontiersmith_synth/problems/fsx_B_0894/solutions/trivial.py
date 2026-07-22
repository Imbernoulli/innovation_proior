# TIER: trivial
import sys, json

inst = json.load(sys.stdin)
N = inst["n"]
P = inst["budget"]

# Split the budget evenly across all lines, decode in the order lines happen
# to be listed in. No attempt at water-filling or interference awareness.
power = [P / N] * N
order = list(range(N))

print(json.dumps({"power": power, "order": order}))
