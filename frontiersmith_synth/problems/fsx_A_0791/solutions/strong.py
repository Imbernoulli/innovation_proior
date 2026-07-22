# TIER: strong
# The insight: price to the demand the REMAINING HORIZON can still absorb,
# not to today's willingness-to-pay. Approximate the shadow value of one more
# unit of remaining stock, against remaining time, with an inventory-PACING
# target: this period should sell at the rate remaining_stock / remaining_
# periods. Solve the period's (known) demand curve A_t(p) = max(0, (a+g*r) -
# (b+g)*p) for the price that hits that pace exactly:
#   p_pace = (a + g*r - target) / (b + g)
# When stock is scarce relative to time left, target is small -> p_pace is
# pulled ABOVE the myopic monopolist price, rationing sales and preserving
# stock for later (higher-a) periods. When stock is abundant relative to
# time left, target is large -> p_pace is pulled DOWN, discounting harder
# than the myopic price to avoid leaving inventory to perish worthless at
# the horizon. This is a genuine reformulation (a fluid/pacing proxy for the
# dual/shadow-price view), not "greedy plus more iterations".
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
p_max = inst["p_max"]
a = inst["a"]; b = inst["b"]; g = inst["g"]
alpha = inst["alpha"]
r = inst["r1"]
inv = inst["C0"]

prices = []
for t in range(T):
    remaining_periods = T - t
    target = inv / remaining_periods
    denom = b[t] + g[t]
    if denom > 0:
        p = (a[t] + g[t] * r - target) / denom
    else:
        p = 0.0
    p = max(0.0, min(p_max, p))
    prices.append(p)

    # advance the true state so later decisions see the real consequence
    A = (a[t] + g[t] * r) - denom * p
    if A < 0.0:
        A = 0.0
    sales = A if A < inv else inv
    inv -= sales
    r = alpha * r + (1.0 - alpha) * p

print(json.dumps({"prices": prices}))
