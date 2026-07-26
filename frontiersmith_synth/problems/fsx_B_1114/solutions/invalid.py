# TIER: invalid
# Always claims to admit every arrival without ever naming an eviction --
# once the list is full this is infeasible and the whole session is
# rejected (score 0), exactly as a garbage/cheating submission should be.
import sys, json

inst = json.load(sys.stdin)
arrivals = inst["arrivals"]
decisions = [{"action": "admit", "evict": None} for _ in arrivals]
print(json.dumps({"decisions": decisions, "state": None}))
