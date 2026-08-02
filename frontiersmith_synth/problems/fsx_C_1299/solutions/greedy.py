# TIER: greedy
# Tight symmetric proportional control: correct every deviation immediately, back to
# slot center, ignoring both the drift's systematic direction and the burn-efficiency
# window. This is the obvious first instinct -- it keeps x tightest to center of any
# policy in this ladder -- but with a near-zero dead-band it fires a correction on
# almost every single step, so it experiences roughly the unconditional average
# efficiency (mostly eff_low) instead of timing burns to eff_high, and burns through
# fuel_budget long before the horizon on every instance.
import sys, json

inst = json.load(sys.stdin)

policy = {"band_lo": 2.0, "band_hi": 2.0, "target_pos": 0.0, "patience": 0}
print(json.dumps(policy))
