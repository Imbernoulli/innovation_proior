# TIER: invalid
# Infeasible regularizer: negative penalties (an "anti-regularizer") are out of the
# declared [0, cap] range, so the evaluator rejects the answer -> score 0 on every instance.
import sys, json
inst = json.load(sys.stdin)
M = inst["M"]
print(json.dumps({"ridge": [-5.0] * M}))
