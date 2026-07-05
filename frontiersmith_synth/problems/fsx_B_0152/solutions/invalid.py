# TIER: invalid
# Hold NO safety stock anywhere. With every k_i = 0 the per-depot backorders are
# maximal (B_i = sd_i * Loss(0) = 0.399 * sd_i), so the network fill rate collapses
# to roughly 0.6 -- well below any beta -- making every instance infeasible. Scores 0
# everywhere.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"stock": [0.0] * inst["N"]}))
