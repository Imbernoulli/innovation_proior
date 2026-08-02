# TIER: trivial
# Do nothing: set both dead-band thresholds far beyond the box, so a correction is
# never due (the 0.95*box_half_width safety net also never engages, since it only
# helps once you're already outside your OWN band -- see statement.md) -- the
# satellite drifts freely until it exits the box on its own. This reproduces the
# evaluator's own weak reference (base_life) exactly, so it scores ~0.1.
import sys, json

inst = json.load(sys.stdin)
W = inst["box_half_width"]

policy = {"band_lo": W * 10.0, "band_hi": W * 10.0, "target_pos": 0.0, "patience": 0}
print(json.dumps(policy))
