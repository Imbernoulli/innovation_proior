# TIER: trivial
# Run no assays at all.  The empty design reproduces the evaluator's baseline
# (full prior variance of theta, zero cost), so it scores ~0.1 on every instance.
import sys, json
json.load(sys.stdin)
print(json.dumps({"probes": []}))
