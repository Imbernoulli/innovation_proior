# TIER: greedy
# Single-key DENSITY greedy: accept packages by descending value-per-licence
# (value / bundle size), which favours efficient small bundles over one huge blocking
# bid.  Better than the value-sorted weak reference on contended instances, worse than
# a multi-start local search.  Price each winner at its bid value.
import sys, json

inst = json.load(sys.stdin)
bids = inst["bids"]
order = sorted(range(len(bids)),
               key=lambda j: (-bids[j]["value"] / len(bids[j]["items"]), j))

used_item = set()
used_bidder = set()
win = []
for j in order:
    bd = bids[j]
    if bd["bidder"] in used_bidder:
        continue
    if any(i in used_item for i in bd["items"]):
        continue
    used_item.update(bd["items"])
    used_bidder.add(bd["bidder"])
    win.append(j)

prices = [bids[j]["value"] for j in win]
print(json.dumps({"win": win, "prices": prices}))
