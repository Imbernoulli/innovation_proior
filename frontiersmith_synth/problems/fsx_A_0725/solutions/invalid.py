# TIER: invalid
# Deliberately breaks the budget contract: dumps far more immunizations into round 0 than
# rate_cap allows, and the total exceeds total_budget too. The evaluator must reject this.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
T = inst["T"]

schedule = [[] for _ in range(T)]
schedule[0] = list(range(min(N, 500)))   # way over rate_cap and total_budget

print(json.dumps({"schedule": schedule}))
