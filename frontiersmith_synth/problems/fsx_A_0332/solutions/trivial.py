# TIER: trivial
# Next-fit porter loading: keep loading the CURRENT load until the next item would
# break either the weight limit or the K-category limit, then open the next load and
# never look back.  Reproduces the evaluator's weak baseline, so it scores ~0.1 on
# every instance.
import sys, json

inst = json.load(sys.stdin)
C = inst["capacity"]
K = inst["classes"]
weights = inst["weights"]
category = inst["category"]

assign = []
b = 0
rem = C
cats = set()
for w, c in zip(weights, category):
    fits_w = (w <= rem)
    fits_c = (c in cats) or (len(cats) < K)
    if fits_w and fits_c:
        rem -= w
        cats.add(c)
        assign.append(b)
    else:
        b += 1
        rem = C - w
        cats = {c}
        assign.append(b)

print(json.dumps({"assign": assign}))
