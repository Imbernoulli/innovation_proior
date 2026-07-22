# TIER: invalid
# Always discharge at full power from step 0, ignoring available state of
# charge. soc0 is only half of capacity0, so by the time the requested
# discharge exceeds the remaining stored energy the trace is infeasible ->
# the evaluator rejects the whole answer and scores it 0.0.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
pmax = inst["power_max"]

print(json.dumps({"actions": [-pmax] * T}))
