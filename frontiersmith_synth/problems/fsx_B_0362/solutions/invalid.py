# TIER: invalid
# Zero-inventory "just-in-time" fantasy: hold nothing anywhere.  Cheapest by
# holding cost, but every failure is an immediate backorder, so the aggregate
# expected backorders blow past the SLA cap -> infeasible -> score 0.
import sys, json
inst = json.load(sys.stdin)
print(json.dumps({"s0": 0, "s": [0] * inst["N"]}))
