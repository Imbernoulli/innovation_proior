# TIER: trivial
# Never charge at all. Every route needs at least one recharge to finish (no
# single leg or half-route alone exceeds capacity, but the whole route does), so
# a van that never plugs in reliably strands partway through and forfeits every
# stop from that point on. This reproduces the evaluator's own weak reference
# construction exactly, so it anchors the score near 0.1.
import sys, json

inst = json.load(sys.stdin)

vans = []
for van in inst["vans"]:
    vans.append({"id": van["id"], "charge_at_p1": 0, "charge_at_p2": 0})

print(json.dumps({"vans": vans}))
