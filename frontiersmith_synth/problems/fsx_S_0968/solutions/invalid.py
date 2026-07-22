# TIER: invalid
# Deliberately broken: overspends the ballast budget on every node AND asks
# for more cargo than the deck's cargo cap allows. Must be rejected (score 0)
# by the evaluator's answer validation.
import sys, json

inst = json.load(sys.stdin)
n = inst["n_nodes"]
budget = inst["ballast_budget"]
cap = inst["cargo_cap"]

ballast = [budget] * n  # sum hugely exceeds ballast_budget
cargo = cap + 1000       # exceeds cargo_cap

print(json.dumps({"ballast": ballast, "cargo": cargo}))
