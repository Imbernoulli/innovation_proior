# TIER: trivial
# Ultra-conservative "do nothing" expedition: submit an empty itinerary. The
# planner never leaves the start room and never risks a single fixture, so it
# banks none of the map bonus and none of the reward -- the mirror-image
# failure mode of blind exhaustion (too timid instead of too reckless).
import sys, json

inst = json.load(sys.stdin)  # unused: this policy never looks at the map
print(json.dumps({"actions": []}))
