# TIER: strong
"""Insight: estimate, from the resolved calibration window, how predictive the
order-flow-imbalance feature is of the next-tick return (a simple regression
slope), then commit a policy that skews quotes BOTH by current inventory
(mean-revert toward flat, sized by realized volatility) AND by the live
order-flow signal in the direction it predicts the price will move -- instead
of waiting to get run over and only then reacting."""
import sys, json

inst = json.load(sys.stdin)
calib = inst["calibration"]
ofi = calib["ofi"]
ret = calib["next_ret"]
n = len(ofi)

if n > 0:
    mean_ofi = sum(ofi) / n
    mean_ret = sum(ret) / n
    cov = sum((ofi[i] - mean_ofi) * (ret[i] - mean_ret) for i in range(n)) / n
    var_ofi = sum((ofi[i] - mean_ofi) ** 2 for i in range(n)) / n
    beta = cov / var_ofi if var_ofi > 1e-9 else 0.0
    var_ret = sum((ret[i] - mean_ret) ** 2 for i in range(n)) / n
    ret_std = var_ret ** 0.5
else:
    beta = 0.0
    ret_std = inst.get("vol_hint", 0.05)

vol_hint = float(inst.get("vol_hint", 0.05))
lo, hi = inst.get("hs_bounds", [0.005, 50.0])

# half spread: sized off the PER-TICK volatility hint (not the multi-tick ret_std
# used for the flow regression above) -- a session-length-independent quote width
# that still covers the typical single-tick noise cost with a small margin.
half_spread = 2.0 * vol_hint + 0.02
half_spread = max(lo, min(hi, half_spread))

# inventory skew: mean-revert toward flat, sized off the same per-tick volatility
# (more volatile regime -> holding inventory is riskier -> skew harder per unit).
inv_coef = max(0.0, min(1e4, 3.0 * vol_hint))

# order-flow skew: move OUT of the way of the direction OFI predicts. `beta` is
# the OLS slope of the ret_horizon-tick-ahead return on the causal OFI feature,
# i.e. it already estimates "expected cumulative price move per unit of current
# order-flow imbalance" over the horizon that matters for this decision. If
# beta>0 (elevated ofi predicts a rise) we want skew<0: both quotes shift UP,
# so our ask is less likely to be undercut by informed buying and our bid
# attracts sellers ahead of the move instead of behind it.
K_OFI = 1.0
ofi_coef = max(-1e4, min(1e4, -K_OFI * beta))

print(json.dumps({"half_spread": half_spread, "inv_coef": inv_coef, "ofi_coef": ofi_coef}))
