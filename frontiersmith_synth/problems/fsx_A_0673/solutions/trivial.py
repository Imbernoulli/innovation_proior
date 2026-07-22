# TIER: trivial
# Do nothing clever: pin no keys, and use plain LRU (w_lru=1, everything else 0).
# This is exactly the evaluator's own weak-baseline reference, so it lands at r~0.1
# on every instance by construction.
import sys, json

inst = json.load(sys.stdin)

print(json.dumps({"pin": [], "w_lru": 1.0, "w_mru": 0.0, "w_lfu": 0.0, "w_scan": 0.0}))
