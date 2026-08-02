# TIER: invalid
# Emits a plausible-looking JSON object that is missing the required "actions"
# list entirely -- the evaluator's answer validator rejects any answer whose
# "actions" field is not a list, so every instance scores 0.0.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"status": "ok", "note": "expedition plan omitted"}))
