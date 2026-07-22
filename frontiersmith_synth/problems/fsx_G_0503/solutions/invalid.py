# TIER: invalid
# Wrong shape: the evaluator requires a dict with finite default_threshold and a
# threshold table.  This must be rejected and score 0.
import json

print(json.dumps({"thresholds": ["ED|low", 0.1]}))
