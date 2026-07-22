# TIER: invalid
# Emits a tour with a duplicate/out-of-range index (repeats city 0 twice and
# never visits the last city) -- not a permutation of 0..n-1, so the
# evaluator's strict validation rejects it -> scores 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
bad_tour = [0] + list(range(0, n - 1))  # duplicate + missing last index
print(json.dumps({"tour": bad_tour}))
