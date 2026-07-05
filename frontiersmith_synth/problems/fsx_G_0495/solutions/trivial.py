# TIER: trivial
# Reproduce the evaluator's WEAK reference: arrival-order (submission-order) greedy.
# Walk the bids in the order they were submitted and accept a package iff all its
# licences are still free and its carrier has not already won -- never reordering,
# never looking back.  Price each winner at its own bid value (individually rational).
# This matches w_base exactly -> ~0.10.
import sys, json

inst = json.load(sys.stdin)
bids = inst["bids"]

used_item = set()
used_bidder = set()
win = []
for j in range(len(bids)):
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
