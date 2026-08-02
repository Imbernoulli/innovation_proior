# TIER: greedy
# The "obvious" textbook move: in a single isolated second-price auction,
# bidding your true estimated value is dominant/optimal, so bid the public
# signal every round, every round. This ignores that (a) the budget is a
# shared, non-replenishing resource across the WHOLE sequence, and (b) the
# competitor watches this bidder's own aggression and escalates. Truthful
# bidding on every low-value round burns budget early AND ratchets the
# competitor up, so by the time the genuinely valuable rounds arrive the
# budget is gone (or clipped) and the competitor is expensive.
import sys, json

inst = json.load(sys.stdin)
base = inst["base"]
print(json.dumps({"bids": [float(x) for x in base]}))
