# TIER: invalid
# Emits a malformed network: an unknown feature name and a non-finite cut point.
# The evaluator's strict validation rejects it -> every instance scores 0.0.
import sys, json

json.load(sys.stdin)
print(json.dumps({"feature": "diffusion_reaction", "smooth": -3, "cuts": [float("nan")]}))
