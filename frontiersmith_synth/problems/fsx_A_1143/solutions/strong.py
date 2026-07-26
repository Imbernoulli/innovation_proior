# TIER: strong
# Same opponent simulation as `greedy`, plus the one genuine insight: since
# every opponent's rule is fully published, a bid on a lot you don't even
# want is never wasted. Because you always lose ties, bidding EXACTLY the
# current top opponent bid is a completely safe "bluff ceiling" -- you never
# win, but the auction's winner now must pay their own full top bid instead
# of the (usually much smaller) natural second-highest price, draining that
# specific opponent's remaining budget for every lot still to come.
#
# So: only actually spend your OWN budget to win a lot when it is a clear
# bargain (value comfortably above the current top bid); on every other lot,
# bluff for free at the safe ceiling instead of walking away for nothing.
# This starves whichever opponent keeps winning the early marquee lots,
# leaving weaker, budget-drained competition on the many ordinary lots later
# in the session -- the same total 300-lot sequence, but priced very
# differently for a buyer who plans against the published rules instead of
# reacting to them lot by lot.
import sys, json

WIN_RATIO = 1.8    # only cash in a lot outright once it beats the top bid by this margin

inst = json.load(sys.stdin)
n = inst["n_lots"]
lots = inst["lots"]
opps = inst["opponents"]          # fixed order: pacer, sniper, capper
pacer, sniper, capper = opps[0], opps[1], opps[2]

cat_max = max(l["estimate"] for l in lots)
rem = {"pacer": pacer["budget"], "sniper": sniper["budget"], "capper": capper["budget"]}
capper_budget0 = capper["budget"]
buyer_rem = inst["buyer_budget"]


def opp_bid(kind, E, remaining):
    if remaining <= 0:
        return 0.0
    if kind == "pacer":
        share = remaining / rem_lots
        raw = pacer["k"] * min(share, E)
    elif kind == "sniper":
        if E >= sniper["thresh"] * cat_max:
            raw = sniper["spike"] * E
        else:
            raw = sniper["low"] * E
    else:  # capper
        raw = min(capper["cap_mult"] * E, capper["cap_frac"] * capper_budget0)
    if raw < 0.0:
        raw = 0.0
    if raw > remaining:
        raw = remaining
    return raw


bids = [0.0] * n
for i in range(n):
    rem_lots = n - i
    E = lots[i]["estimate"]
    v = lots[i]["value"]
    bp = opp_bid("pacer", E, rem["pacer"])
    bs = opp_bid("sniper", E, rem["sniper"])
    bc = opp_bid("capper", E, rem["capper"])
    top = max(bp, bs, bc)

    if top <= 1e-9:
        # nothing left to bluff or beat -- take the lot for free if it has any value
        bid = 0.01 if v > 0.0 else 0.0
    elif v > WIN_RATIO * top and top + 0.01 <= buyer_rem:
        bid = top + 0.01           # a clear bargain: cash in now, spend our own budget
    else:
        bid = top                  # safe bluff ceiling: drain the leading opponent for free

    bids[i] = bid

    entries = [("pacer", bp, 0), ("sniper", bs, 1), ("capper", bc, 2), ("buyer", bid, 3)]
    entries.sort(key=lambda t: (-t[1], t[2]))
    winner, price = entries[0][0], entries[1][1]
    if winner == "buyer":
        buyer_rem -= price
    else:
        rem[winner] -= price

print(json.dumps({"bids": bids}))
