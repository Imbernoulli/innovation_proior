# TIER: strong
# Multi-key greedy start + 1-for-many-swap local search.  Try several package
# orderings (value, density, sqrt-density, small-first), greedily allocate each, then
# repeatedly improve: for any package not yet winning, drop the winners it collides
# with (licence or same-carrier XOR) and adopt it whenever that strictly raises
# welfare.  Keep the best plan over all starts.  Lands near the internal strong anchor
# (~0.8) while leaving headroom for a truly optimal winner-determination solver.
import sys, json

inst = json.load(sys.stdin)
bids = inst["bids"]
n = len(bids)


def greedy(order):
    used_item = set(); used_bidder = set(); win = []
    for j in order:
        bd = bids[j]
        if bd["bidder"] in used_bidder:
            continue
        if any(i in used_item for i in bd["items"]):
            continue
        used_item.update(bd["items"]); used_bidder.add(bd["bidder"]); win.append(j)
    return win


def welfare(win):
    return sum(bids[j]["value"] for j in win)


def local_search(win, passes):
    winset = set(win)
    item_owner = {}; bidder_owner = {}
    for j in win:
        for i in bids[j]["items"]:
            item_owner[i] = j
        bidder_owner[bids[j]["bidder"]] = j

    def remove(j):
        winset.discard(j)
        for i in bids[j]["items"]:
            if item_owner.get(i) == j:
                del item_owner[i]
        if bidder_owner.get(bids[j]["bidder"]) == j:
            del bidder_owner[bids[j]["bidder"]]

    def add(j):
        winset.add(j)
        for i in bids[j]["items"]:
            item_owner[i] = j
        bidder_owner[bids[j]["bidder"]] = j

    for _ in range(passes):
        improved = False
        for j in range(n):
            if j in winset:
                continue
            bd = bids[j]
            conflicts = set()
            bo = bidder_owner.get(bd["bidder"])
            if bo is not None:
                conflicts.add(bo)
            for i in bd["items"]:
                o = item_owner.get(i)
                if o is not None:
                    conflicts.add(o)
            if bd["value"] - sum(bids[c]["value"] for c in conflicts) > 0:
                for c in list(conflicts):
                    remove(c)
                add(j)
                improved = True
        if not improved:
            break
    return sorted(winset)


keys = [
    lambda j: (-bids[j]["value"], j),
    lambda j: (-bids[j]["value"] / len(bids[j]["items"]), j),
    lambda j: (-bids[j]["value"] / (len(bids[j]["items"]) ** 0.5), j),
    lambda j: (len(bids[j]["items"]), -bids[j]["value"], j),
]

best_win, best_w = None, -1
for key in keys:
    win = local_search(greedy(sorted(range(n), key=key)), passes=6)
    w = welfare(win)
    if w > best_w:
        best_w, best_win = w, win

prices = [bids[j]["value"] for j in best_win]
print(json.dumps({"win": best_win, "prices": prices}))
