# TIER: trivial
# Predict NO causal links at all.  The empty graph is the evaluator's weak
# reference (its SHD equals the number of true edges), so this scores ~0.1 on
# every region.  A "learn nothing" baseline.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"edges": []}))
