# TIER: greedy
# The obvious "professional" recipe: every period, post the price that
# maximizes THAT period's own expected revenue given the live demand curve
# and the CURRENT reference level (classic single-period monopolist price for
# A_t(p) = max(0, (a+g*r) - (b+g)*p), maximized at p* = (a+g*r) / (2*(b+g))).
# It reacts to the real, changing demand schedule each period -- it is NOT
# a frozen guess -- but it never looks ahead: it does not know or care how
# much inventory remains or how many periods are left, so on scarce-stock or
# back-loaded-demand instances it happily sells out early at bargain prices
# and has nothing left to offer the richer late crowd.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
p_max = inst["p_max"]
a = inst["a"]; b = inst["b"]; g = inst["g"]
alpha = inst["alpha"]
r = inst["r1"]

prices = []
for t in range(T):
    denom = 2.0 * (b[t] + g[t])
    p = (a[t] + g[t] * r) / denom if denom > 0 else 0.0
    p = max(0.0, min(p_max, p))
    prices.append(p)
    r = alpha * r + (1.0 - alpha) * p

print(json.dumps({"prices": prices}))
