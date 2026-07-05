# TIER: trivial
# Roster order: inspect the lift stations in exactly the order they appear on the
# maintenance roster, i.e. the identity permutation [1, 2, ..., N].  This is the
# evaluator's do-nothing baseline circuit, so it scores exactly ~0.1.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
print(json.dumps({"order": list(range(1, n + 1))}))
