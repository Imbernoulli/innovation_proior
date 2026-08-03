# TIER: invalid
# Wrong-shape answer (a constant scalar instead of a length-K activation curve) -> the
# evaluator's strict validator rejects it and every site scores 0.
import sys, json
json.load(sys.stdin)
print(json.dumps(0.0))
