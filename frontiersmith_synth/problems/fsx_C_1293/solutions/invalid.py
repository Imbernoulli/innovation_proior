# TIER: invalid
# Claims "orders" is a bare string instead of a list -- fails the top-level
# schema check on turn 0, so the entire battle is rejected (score 0), the
# same way any garbage/cheating submission should be.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"orders": "gimme_the_win", "state": None}))
