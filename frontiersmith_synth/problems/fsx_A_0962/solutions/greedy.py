# TIER: greedy
# The obvious "product pricing" approach: treat every region as its own isolated
# monopoly and set its own revenue-maximizing price assuming v_i = base_i, i.e.
# pretending no one else will ever adopt.
#
# This is the trap. It never looks at the influence network W at all. A region
# with modest local value but huge downstream influence gets priced for its own
# sake, not for what its adoption is worth to the regions that depend on it --
# so the network-wide cascade that a coordinated price could ignite never fires.
import sys, json, math


def sigmoid(z):
    if z > 40.0:
        return 0.0
    if z < -40.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def best_price_for(v, spread, steps=250):
    hi = v * 3 + 50
    best_p, best_r = 0.0, -1.0
    for k in range(steps + 1):
        p = hi * k / steps
        r = p * sigmoid((p - v) / spread)
        if r > best_r:
            best_r, best_p = r, p
    return best_p


inst = json.load(sys.stdin)
base, spread = inst["base"], inst["spread"]
prices = [best_price_for(base[i], spread[i]) for i in range(inst["m"])]
print(json.dumps({"prices": prices}))
