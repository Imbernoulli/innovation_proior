# TIER: invalid
# Award EVERY package bid to its carrier.  Overlapping footprints share licences and
# most carriers own several XOR packages, so this violates single-assignment and XOR
# feasibility on any non-trivial instance -> rejected -> scores 0.0.
import sys, json

inst = json.load(sys.stdin)
bids = inst["bids"]
win = list(range(len(bids)))
prices = [bids[j]["value"] for j in win]
print(json.dumps({"win": win, "prices": prices}))
