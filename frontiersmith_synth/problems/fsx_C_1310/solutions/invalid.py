# TIER: invalid
# Deliberately broken: lists the SAME target id in both pass1 and pass2 (a target
# can be attempted only once total), which must be rejected by validation and
# score 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
ids = [t["id"] for t in inst["targets"]]
dup = ids[0] if ids else 0
print(json.dumps({"pass1": [dup], "pass2": [dup]}))
