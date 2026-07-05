# TIER: invalid
# Cram every contact into a single pod.  For every instance in this family the
# total load exceeds one pod's capacity C (and the number of distinct lineages
# exceeds K), so pod 0 violates BOTH constraints -> the layout is infeasible ->
# the evaluator scores it 0.0.
import sys, json

inst = json.load(sys.stdin)
N = inst["n"]

print(json.dumps({"assign": [0] * N}))
