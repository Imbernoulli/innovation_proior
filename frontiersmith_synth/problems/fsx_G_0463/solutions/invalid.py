# TIER: invalid
# Emit a structurally broken tree: the root is an internal node whose children point
# to indices that do not exist (only one node is present).  Validation rejects the
# out-of-range child index, so the evaluator scores this 0.0.
import sys, json

json.load(sys.stdin)
print(json.dumps({"nodes": [{"feature": 0, "threshold": 0.0, "left": 7, "right": 9}]}))
