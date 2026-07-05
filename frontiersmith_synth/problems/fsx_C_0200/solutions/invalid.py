# TIER: invalid
# Emit a CYCLIC "influence map" (0->1->2->0).  A causal DAG must be acyclic,
# so the evaluator rejects this as infeasible and scores it 0 on every
# instance -- demonstrates the feasibility gate.
import sys, json
inst = json.load(sys.stdin)
d = inst["d"]
if d >= 3:
    edges = [[0, 1], [1, 2], [2, 0]]
else:
    edges = [[0, 1], [1, 0]]
print(json.dumps({"edges": edges}))
