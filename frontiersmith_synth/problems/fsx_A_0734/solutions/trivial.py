# TIER: trivial
# "Always give a big, generous coupon" -- the naive volume-chasing instinct. Applies a
# deep flat 37% discount every single week, to every segment, forever, never looking
# at the pilot data at all. Reference price converges to the deeply discounted price
# almost immediately, the "bargain" gap vanishes for good, and the segment is then
# sold to at a permanently thin margin for the rest of the year.
import sys, json

FLAT_DEPTH = 0.37

inst = json.load(sys.stdin)
n_weeks = inst["n_weeks"]
dmax = inst["max_discount"]
m = len(inst["segments"])

depth = min(FLAT_DEPTH, dmax)
schedule = [[depth] * n_weeks for _ in range(m)]
print(json.dumps({"schedule": schedule}))
