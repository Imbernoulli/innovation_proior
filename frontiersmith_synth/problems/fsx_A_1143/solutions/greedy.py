# TIER: greedy
# Honest, opponent-aware pacer -- the obvious first algorithm. It DOES read
# and faithfully simulate the three published opponent formulas (you have to,
# just to know whether a bid would win), and it correctly tracks its own
# remaining budget lot by lot. But it only ever bids on a lot it actually
# wants: if the top opponent bid it computes is below its own value, it bids
# just enough to win; otherwise it bids 0 and walks away. It never places a
# bid on a lot it does NOT want, so it never spends a single opponent's money
# for free -- every marquee "decoy" lot with a low value to this buyer is
# simply skipped, leaving the sniper/pacer/capper's budgets fully intact for
# every lot that follows.
import sys, json

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

    bid = 0.0
    if v > top and top + 0.01 <= buyer_rem:
        bid = top + 0.01           # bid just enough to win a lot we actually want

    bids[i] = bid

    # advance state to keep budget bookkeeping correct for later lots
    entries = [("pacer", bp, 0), ("sniper", bs, 1), ("capper", bc, 2), ("buyer", bid, 3)]
    entries.sort(key=lambda t: (-t[1], t[2]))
    winner, price = entries[0][0], entries[1][1]
    if winner == "buyer":
        buyer_rem -= price
    else:
        rem[winner] -= price

print(json.dumps({"bids": bids}))
