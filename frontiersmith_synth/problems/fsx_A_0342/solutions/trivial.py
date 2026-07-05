# TIER: trivial
# Next-fit dispatch: keep loading the current transformer until a block does not
# fit thermally OR the breaker panel is full, then energize the next transformer
# and never look back.  Reproduces the evaluator's weak baseline, so it scores
# ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
K = inst["channels"]
demands = inst["demands"]

assign = []
t = 0
rem = C
cnt = 0
for d in demands:
    if d <= rem and cnt < K:
        assign.append(t)
        rem -= d
        cnt += 1
    else:
        t += 1
        rem = C - d
        cnt = 1
        assign.append(t)

print(json.dumps({"assign": assign}))
