# TIER: greedy
# The obvious first pass: trade every time the price looks attractive versus
# the series' own average, at FULL power, ignoring how the instance's own
# eta_c/eta_d/aging_coeff/degradation_price shape the true breakeven spread,
# and with no notion of shallow-cycling or reserving capacity for later
# swings. A fixed +/-8% margin around the mean is used as the "spread looks
# wide enough" heuristic -- reasonable-sounding, but it fires on lots of
# ordinary noise (each trade taxed by round-trip efficiency loss) and always
# dumps the full power limit in one step (deep single-step discharges pay
# the quadratic aging penalty hard), so by the time a real seeded swing
# arrives the battery is often capacity-degraded and mis-positioned.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
prices = inst["prices"]
cap0 = inst["capacity0"]
pmax = inst["power_max"]
soc0 = inst["soc0"]
eta_c = inst["eta_c"]
eta_d = inst["eta_d"]
aging_coeff = inst["aging_coeff"]

mean_p = sum(prices) / len(prices)
buy_thresh = mean_p * 0.92
sell_thresh = mean_p * 1.08

# Greedy DOES track soc/capacity correctly (any competent coder avoids
# crashing into an infeasible trade) -- it just never stops to ask whether
# the *threshold* it trades on actually clears the instance's own aging +
# efficiency breakeven, and never shallow-cycles: full power, every time.
soc = soc0
cap = float(cap0)
actions = [0.0] * T
for t in range(T):
    p = prices[t]
    if p <= buy_thresh:
        room = (cap - soc) / eta_c if eta_c > 1e-9 else 0.0
        x = min(pmax, max(0.0, room))
        if x > 1e-9:
            actions[t] = x
            soc = min(soc + x * eta_c, cap)
    elif p >= sell_thresh:
        d = min(pmax, soc)
        if d > 1e-9:
            actions[t] = -d
            dod = d / cap if cap > 1e-9 else 0.0
            fade = aging_coeff * cap0 * dod * dod
            cap = max(cap - fade, 0.05 * cap0)
            soc -= d

print(json.dumps({"actions": actions}))
