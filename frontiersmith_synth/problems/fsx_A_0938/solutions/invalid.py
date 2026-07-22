# TIER: invalid
"""Malformed on purpose: some table values are not in {STAY,N,E,S,W} (e.g. a diagonal
"NE" and a numeric code), so the evaluator's strict schema check must reject the WHOLE
answer -> score 0 on every instance."""
import sys, json

json.load(sys.stdin)
table = {"5555": "NE", "0000": "STAY", "8888": 1, "7777": "teleport"}
print(json.dumps({"table": table, "default": "STAY"}))
