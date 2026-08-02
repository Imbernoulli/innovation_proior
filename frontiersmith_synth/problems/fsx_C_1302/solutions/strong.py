# TIER: strong
# Insight: pace against the ARRIVAL distribution of value, not against the
# budget clock. Instead of asking "what is this round worth?" round by round
# (the greedy trap), first price out what EVERY round would cost to win if
# the competitor were fully escalated (its worst case, `comp_base*(1+adapt)`
# -- computable in advance from public data alone), then treat the whole
# sequence as a BUDGETED KNAPSACK: rank rounds by surplus PER DOLLAR of that
# worst-case cost (not by raw value), and greedily fund the highest-density
# rounds first until the budget runs out. Rounds that don't make the cut are
# skipped entirely (bid 0) -- this both saves budget AND keeps the
# competitor calm (its adaptation signal is the bidder's own aggression), so
# the escalated-cost estimate used for funded rounds is usually pessimistic
# and the realized price is cheaper than budgeted. A funded round is bid at
# a small (12%) margin over its worst-case cost -- second-price billing
# means the margin never raises what is actually PAID, it only guarantees
# the round is won.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
base = inst["base"]
comp_base = inst["comp_base"]
adapt_rate = inst["adapt_rate"]
budget = inst["budget"]

costs = [comp_base[i] * (1.0 + adapt_rate) for i in range(T)]
idx = [i for i in range(T) if base[i] > costs[i] and costs[i] > 1e-9]
idx.sort(key=lambda i: (base[i] - costs[i]) / costs[i], reverse=True)

bids = [0.0] * T
spent = 0.0
for i in idx:
    c = costs[i] * 1.12
    if spent + c <= budget:
        bids[i] = c
        spent += c

print(json.dumps({"bids": bids}))
