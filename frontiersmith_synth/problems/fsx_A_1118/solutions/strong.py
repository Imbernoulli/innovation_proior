# TIER: strong
# Insight: the right acceptance threshold is a SHADOW PRICE on remaining
# capacity, not one static density cutoff -- and it must also adapt to how the
# real stream's value drift departs from what the preview implied.
#
#   1. Dual-fitting anchor.  Compute lambda* from the preview: sort preview
#      lots by density descending and fill the (shared, known) capacity; the
#      density at which the cumulative size first crosses capacity is the
#      fractional-relaxation cutoff -- the correct STATIC dual price for the
#      load level the preview implies.  This alone already beats a naive mean
#      -density cutoff, which ignores how scarce the capacity actually is.
#   2. Capacity-shadow-price ramp.  Set cap_gain > 0 so the threshold climbs
#      (quadratically, per the evaluator's replay formula) as capacity is
#      actually consumed.  If today runs busier than the preview implied, the
#      bar rises automatically, preserving room for the better lots still to
#      come -- something a single fixed cutoff can never do.
#   3. Drift correction.  time_gain is set to CANCEL the "on-pace" component
#      of the capacity ramp (so a stream that consumes capacity exactly in
#      step with elapsed time sees a flat bar) and then ADD a directional
#      nudge from the preview's own first-half vs second-half density trend,
#      so a rising trend raises the late-stream bar and a falling trend does
#      not over-penalize the (already-good) early lots. drift_gain gives a
#      small further reaction to the observed running-average density level.
#
# This is a genuine reformulation (dual price + drift term), not "greedy with
# more iterations": greedy never even considers capacity state or trend.
import sys, json

inst = json.load(sys.stdin)
capacity = inst["capacity"]
preview = inst["preview"]

# --- 1. lambda*: fractional-knapsack cutoff density from the preview ---
order = sorted(range(len(preview)), key=lambda i: preview[i][1] / preview[i][0], reverse=True)
rem = capacity
lam = None
for i in order:
    size, value = preview[i]
    d = value / size
    lam = d
    if size <= rem:
        rem -= size
    else:
        break
if lam is None:
    lam = 1.0

# --- 2 & 3. drift trend from preview (first half vs second half density) ---
n = len(preview)
half = max(1, n // 2)
densities = [v / s for s, v in preview]
avg_first = sum(densities[:half]) / half
avg_second = sum(densities[half:]) / max(1, n - half)
slope = (avg_second - avg_first) / max(1e-9, avg_first)
slope_clamped = max(-1.0, min(2.0, slope))

B_SCALE, C_SCALE, T_SCALE, D_SCALE = 1.0, 0.5, 0.6, 0.1

base = lam * B_SCALE
cap_gain = lam * C_SCALE
time_gain = -cap_gain + lam * T_SCALE * slope_clamped
drift_gain = D_SCALE

policy = {"base": base, "cap_gain": cap_gain, "drift_gain": drift_gain, "time_gain": time_gain}
print(json.dumps({"policy": policy}))
