# TIER: invalid
# Omits the required "patience" field from the policy, so evaluator._validate rejects
# it -- every instance scores 0.0.
import sys, json

inst = json.load(sys.stdin)
W = inst["box_half_width"]

policy = {"band_lo": 0.5 * W, "band_hi": 0.5 * W, "target_pos": 0.0}
print(json.dumps(policy))
