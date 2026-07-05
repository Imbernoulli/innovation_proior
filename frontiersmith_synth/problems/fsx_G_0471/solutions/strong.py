# TIER: strong
# State-aware curriculum: the training dynamics are fully specified in the public
# instance, so simulate the student ourselves and, at every step, show the example
# whose update reduces the loss the MOST right now:
#     gain(i) = LR * readiness_i * (1 - m_{concept_i})
# This adaptively front-loads foundations exactly until they unlock their
# dependents, then chases whichever concept is currently the cheapest to advance
# -- a myopic greedy on the real state that beats any fixed easy-to-hard order.
# It still pays the dependency tax (readiness < 1 while prerequisites mature), so
# it stays well short of the prerequisite-free optimistic bound: headroom remains.
import sys, json

inst = json.load(sys.stdin)
N = inst["n_examples"]
K = inst["K"]
cap = inst["cap"]
LR = inst["LR"]
target = inst["target"]
ex = inst["examples"]

concept = [ex[i]["concept"] for i in range(N)]
prereqs = [ex[i]["prereqs"] for i in range(N)]

m = [0.0] * K
schedule = []
for _ in range(cap):
    best = -1
    best_gain = -1.0
    for i in range(N):
        P = prereqs[i]
        readiness = 1.0 if not P else min(m[p] for p in P)
        g = LR * readiness * (1.0 - m[concept[i]])
        if g > best_gain:
            best_gain = g
            best = i
    P = prereqs[best]
    readiness = 1.0 if not P else min(m[p] for p in P)
    c = concept[best]
    m[c] += LR * readiness * (1.0 - m[c])
    schedule.append(best)
    loss = sum(1.0 - x for x in m) / K
    if loss <= target:
        break

print(json.dumps({"schedule": schedule}))
