# TIER: strong
# Full inverse-frequency (class-balanced) reweighting: weight each patient by
# N / (K * count_of_its_class), so every class contributes an equal total training
# mass.  This is the textbook fix for macro-F1 under heavy imbalance -- it fully
# equalizes the healthy majority and the two rare subtypes, so the fixed classifier
# stops collapsing onto "predict healthy" and recovers rare-class recall.
# It cannot reach the balanced-ideal ceiling (only 5-22 rare patients exist, far
# fewer than the ideal cohort), so it leaves genuine headroom above it -- e.g. for
# a policy that also synthesizes / jitters minority points -- but among simple
# reweighting rules it is the strong one.
import sys, json

inst = json.load(sys.stdin)
y = inst["y"]
counts = inst["class_counts"]
K = inst["n_classes"]
N = inst["n"]
w = [N / (K * counts[c]) for c in y]
print(json.dumps({"weights": w}))
