# TIER: invalid
# "Just run every assay in the catalogue."  The total cost of all assays far
# exceeds the budget, so the design is infeasible -> the evaluator scores it 0.0.
# A classic budget-blind mistake.
import sys, json
inst = json.load(sys.stdin)
print(json.dumps({"probes": list(range(inst["M"]))}))
