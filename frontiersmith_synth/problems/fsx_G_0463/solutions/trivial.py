# TIER: trivial
# "No split" criterion: never grow the tree, just predict the majority class of the
# training set from a single leaf.  This reproduces the evaluator's majority-class
# baseline, so it scores ~0.1.  It captures none of the population's risk regions.
import sys, json

inst = json.load(sys.stdin)
y = inst["y_train"]
c1 = sum(y); c0 = len(y) - c1
maj = 0 if c0 >= c1 else 1
print(json.dumps({"nodes": [{"leaf": maj}]}))
