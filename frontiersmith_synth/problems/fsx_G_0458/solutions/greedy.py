# TIER: greedy
# Square-root inverse-frequency reweighting: weight each patient by 1/sqrt(count of
# its class).  This is a common "gentle" rebalancing heuristic -- it nudges the
# rare subtypes up without letting a handful of rare patients dominate training.
# It clearly beats doing nothing, but it UNDER-corrects the imbalance, so it leaves
# a lot of macro-F1 on the table versus fuller reweighting.
import sys, json, math

inst = json.load(sys.stdin)
y = inst["y"]
counts = inst["class_counts"]
w = [1.0 / math.sqrt(counts[c]) for c in y]
print(json.dumps({"weights": w}))
