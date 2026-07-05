# TIER: greedy
# Foundation-first curriculum by static difficulty: sort the examples by their
# concept layer (shallow prerequisites before deep dependents) and cycle that
# fixed easy-to-hard order.  Because a concept's prerequisites all live in
# earlier layers, every example is shown only after (an example of) its
# prerequisites has appeared earlier in the same pass, so the dependency chain
# lights up far faster than the shuffled as-shipped order -- but it ignores the
# actual training state, so it wastes updates on already-mastered concepts.
import sys, json

inst = json.load(sys.stdin)
N = inst["n_examples"]
cap = inst["cap"]
ex = inst["examples"]

order = sorted(range(N), key=lambda i: ex[i]["layer"])
schedule = [order[i % N] for i in range(cap)]
print(json.dumps({"schedule": schedule}))
