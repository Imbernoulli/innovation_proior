# TIER: invalid
# Malformed on purpose: "masses" contains a non-numeric string and a
# negative value, and the arrays are the wrong length. Must score 0.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"routes": [0, 1], "masses": ["lots", -5.0]}))
