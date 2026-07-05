# TIER: invalid
# Dumps every artifact into a single crate, ignoring mass capacity and slot limit.
# This overfills crate 0 on essentially all instances -> rejected -> 0.0.
import sys, json

inst = json.load(sys.stdin)
N = inst["n"]
print(json.dumps({"assign": [0] * N}))
