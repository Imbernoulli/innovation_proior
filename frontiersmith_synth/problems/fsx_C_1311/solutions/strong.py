# TIER: strong
"""The insight: convert the stated forecast-uncertainty band into a rolling,
PRE-EMPTIVE reserve, sized to a forward-looking window rather than a single
constant. For each day t, build a pessimistic PV estimate over the next W days
(forecast - K*uncertainty, clipped at 0) and ask "how much extra battery energy
would I need, starting today, to still cover the window's CRITICAL demand under
that pessimistic PV path?" -- that shortfall becomes today's reserve floor for
'important'. Adding the window's IMPORTANT demand to that same shortfall gives a
(necessarily larger) floor for 'low', so low tier sheds first, important sheds
second, and critical is never touched by the policy at all.

Because the pessimistic estimate is small only where the stated uncertainty is
actually elevated, this reserve stays ~0 on calm days (capturing full value,
same as greedy) but ramps up several days BEFORE a signalled low-sun sequence
hits -- exactly wide enough to survive it -- then relaxes again afterward. A
single constant reserve (the best any non-adaptive policy could pick per
instance) cannot do this: it either wastes value on calm days or arrives too
late/too small for the sequence. This is not "greedy plus tuning" -- it is a
genuine reformulation of forecast uncertainty into a time-varying feasibility
reserve."""
import sys, json

W = 4       # lookahead window (days) -- matches the theme's "forecast-uncertainty window"
K = 1.4     # pessimism multiplier on the stated per-day uncertainty


def main():
    inst = json.load(sys.stdin)
    T = int(inst["T"])
    cap = float(inst["capacity_kwh"])
    forecast = [float(x) for x in inst["pv_forecast_kwh"]]
    unc = [float(x) for x in inst["pv_forecast_uncertainty_kwh"]]
    crit = [float(x) for x in inst["critical_demand_kwh"]]
    imp = [float(x) for x in inst["important_demand_kwh"]]

    res_imp = [0.0] * T
    res_low = [0.0] * T
    for t in range(T):
        hi = min(T, t + W)
        lo_pv = [max(0.0, forecast[s] - K * unc[s]) for s in range(t, hi)]
        win_crit = sum(crit[t:hi])
        win_imp = sum(imp[t:hi])
        win_pv = sum(lo_pv)
        res_imp[t] = min(cap, max(0.0, win_crit - win_pv))
        res_low[t] = min(cap, max(0.0, win_crit + win_imp - win_pv))

    ans = {"reserve_important_kwh": res_imp, "reserve_low_kwh": res_low}
    print(json.dumps(ans))


if __name__ == "__main__":
    main()
