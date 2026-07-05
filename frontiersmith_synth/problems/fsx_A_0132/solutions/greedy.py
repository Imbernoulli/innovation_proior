# TIER: greedy
# First-fit priority: give every fitting module the same constant score (phi0 = 1),
# so the lowest-index module that still has room wins the tie. This reuses gaps in
# earlier modules instead of spreading load like worst-fit, powering fewer modules,
# but it never prefers the tightest module so it leaves easy consolidations on the
# table.
import sys, json

json.load(sys.stdin)
print(json.dumps({"weights": [1.0, 0.0, 0.0, 0.0]}))
