# TIER: strong
# Asymmetric dead-band sized from the drift's systematic direction and rate, timed to
# the burn-efficiency window.
#   1) drift_rate = |bias| + noise_amp/2 estimates how fast x moves per step in the
#      worst case. safety_margin = drift_rate * period * 1.15 is the distance x could
#      cover while waiting a full efficiency cycle for the next eff_high window.
#   2) The threshold on the side the drift PUSHES TOWARD is pulled in by exactly that
#      safety_margin (band_near = box_half_width - safety_margin), so once a
#      correction is due there is still room left to wait for an efficient window
#      before the 0.95*box_half_width safety burn would have to fire. The threshold on
#      the side AWAY from the drift is left almost at the box edge (band_far =
#      0.95*box_half_width), since the systematic push won't carry x there.
#   3) Each correction recenters not to slot center but to target_pos = the far side,
#      well away from the direction of drift, maximizing the distance (and hence
#      steps) until the next correction is due -- fewer, larger burns instead of many
#      small ones, and (2) buys the time to fire them inside the efficient window.
#   4) patience is derived, not guessed: the number of steps available between the
#      near threshold and the safety margin, divided by drift_rate, capped at one
#      full period (waiting longer never helps -- the window recurs every period).
# This directly beats "correct every deviation immediately" (the greedy trap): fewer,
# larger, well-timed burns spend far less fuel per unit of drift corrected.
import sys, json

inst = json.load(sys.stdin)
W = inst["box_half_width"]
b = inst["bias"]
A = inst["noise_amp"]
P = inst["period"]

drift_rate = abs(b) + 0.5 * A
if drift_rate < 1e-9:
    drift_rate = 1e-9

safety_margin = drift_rate * P * 1.15
band_far = 0.95 * W
band_near = W - safety_margin
band_near = max(0.15 * W, min(0.85 * W, band_near))

urgent_room = 0.95 * W - band_near
patience = int(urgent_room / drift_rate)
patience = max(0, min(P, patience))

if b > 1e-9:
    band_hi, band_lo, target = band_near, band_far, -0.65 * W
elif b < -1e-9:
    band_lo, band_hi, target = band_near, band_far, 0.65 * W
else:
    band_hi, band_lo, target = 0.85 * W, 0.85 * W, 0.0

policy = {"band_lo": band_lo, "band_hi": band_hi, "target_pos": target, "patience": patience}
print(json.dumps(policy))
