# TIER: invalid
# Prescribe an out-of-range relaxation factor (and a non-finite one): the
# evaluator's validation rejects it -> score 0.
import sys, json
json.load(sys.stdin)
print(json.dumps({"omega": 7.5, "aitken_period": 3}))
