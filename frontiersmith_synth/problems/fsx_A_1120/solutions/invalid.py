# TIER: invalid
# Outputs a fixed, tiny move list that essentially never restores every book
# to its home (every instance needs a nonzero, instance-specific hall/shelf
# correction). The evaluator checks the final state equals the identity
# permutation exactly; this answer fails that check on every instance, so it
# scores 0.0 throughout.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"moves": ["H+", "S0+"]}))
