# TIER: greedy
# Textbook reactive LRU/LFU cache replayed over the given request table: after
# "observing" each step's traffic, admit the C items that just drew the most
# weight -- i.e. cache_seq[t] = top C items by weight[t]. This is the obvious
# "keep whatever is currently popular" recipe an average coder reaches for first,
# and it is exactly how a real online cache behaves (update admission from what
# you just served). The bug: because a newly-admitted item only starts serving
# traffic from the NEXT step onward (one-step fetch latency -- see statement),
# admitting based on weight[t] can defend step t+1 at best. It never looks past
# what has already happened, so on a fast-moving or wrapping drift it is
# perpetually a step (or more) behind the hot block's leading edge: it keeps
# missing ids just as they turn hot, and it churns -- evicting ids the moment
# they cool off, only to pay a fresh fetch when the drift cycles back to them.
import sys, json

inst = json.load(sys.stdin)
M = inst["M"]; T = inst["T"]; C = inst["C"]
W = inst["weight"]

cache = []
for t in range(T):
    row = W[t]
    order = sorted(range(M), key=lambda i: (-row[i], i))
    cache.append(order[:C])

print(json.dumps({"cache": cache}))
