# TIER: invalid
# Tries to cache the entire catalog every step (no eviction, no capacity
# awareness). Since the catalog size M is always chosen bigger than the cache
# capacity C, this violates the hard per-step size cap on EVERY instance and is
# rejected as infeasible -> scores 0 everywhere.
import sys, json

inst = json.load(sys.stdin)
M = inst["M"]; T = inst["T"]
cache = [list(range(M)) for _ in range(T)]
print(json.dumps({"cache": cache}))
