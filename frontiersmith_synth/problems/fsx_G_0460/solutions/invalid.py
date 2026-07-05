# TIER: invalid
# Over-budget policy: request far more augmented copies than the advertised
# total_copies cap allows (8 + 8 = 16 > 10).  The evaluator rejects any policy whose
# summed copy count exceeds the budget, so this is infeasible -> scores 0.0 on every
# instance.
import sys, json

json.load(sys.stdin)
policy = {"ops": [{"type": "shift", "mag": 3, "copies": 8},
                  {"type": "noise", "std": 0.2, "copies": 8}]}
print(json.dumps(policy))
