# TIER: invalid
"""Malformed output: wrong shape entirely (a list, not the required object) so
the evaluator's strict validation must reject it -> score 0 on every instance."""
import sys, json

json.load(sys.stdin)
print(json.dumps([0.06, 0.0, 0.0]))
