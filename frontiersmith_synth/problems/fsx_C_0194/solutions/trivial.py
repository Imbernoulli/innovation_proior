# TIER: trivial
# Predict NO edges at all -- the empty propagation graph. This is a valid DAG,
# so it is always feasible, and its SHD equals the number of true edges, which
# is exactly the evaluator's calibration baseline -> score ~= 0.1 everywhere.
import sys, json
json.load(sys.stdin)
print(json.dumps({"edges": []}))
