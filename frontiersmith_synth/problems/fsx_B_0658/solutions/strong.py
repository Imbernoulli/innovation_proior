# TIER: strong
# Pilot-probe, then Neyman-allocate:
#   1. Estimate each region's noise std sigma_hat_r from its pilot sample
#      (sample standard deviation, Bessel-corrected).
#   2. Stopping-rule hedge: a handful of pilot readings is itself a noisy
#      witness of the true sigma_r (with pilot_size=5 the sampling
#      distribution of the estimate is wide). Never let a region's
#      estimate collapse to near-zero purely from a lucky quiet pilot --
#      floor every estimate at a fraction of the cross-region median
#      before deciding to effectively stop refining it.
#   3. Neyman allocation: split the budget proportional to
#      width_r * sigma_eff_r (largest width-times-uncertainty product
#      gets the most budget), which is exactly what minimizes
#      sum_r w_r^2*sigma_r^2/(pilot_size+alloc_r) subject to a fixed
#      total budget.
import sys, json, statistics

inst = json.load(sys.stdin)
regions = inst["regions"]
R = len(regions)
B = inst["budget"]

ests = []
for reg in regions:
    pilot = reg["pilot"]
    if len(pilot) > 1:
        s = statistics.stdev(pilot)
    else:
        s = 1.0
    ests.append(max(s, 1e-9))

med = statistics.median(ests) if ests else 1.0
floor = 0.15 * med
eff = [max(e, floor) for e in ests]

weights = [regions[r]["width"] * eff[r] for r in range(R)]
tot = sum(weights)
if tot <= 0:
    alloc = [B / R] * R
else:
    alloc = [B * w / tot for w in weights]

print(json.dumps({"alloc": alloc}))
