# TIER: strong
# The genuine insight: capacity is a consumable spread across BOTH corridors
# and time, not a single "best path" to exploit every day. This solver
# simulates all four corridors' thickness and fatigue exactly (everything
# needed is published in the input), and every day picks the corridor that
# maximizes a MARGINED safe mass (never the exact edge -- leaving fatigue
# headroom so today's crossing doesn't fully cancel tomorrow's decay) plus a
# small bonus for corridors that have been resting, which naturally rotates
# load away from whichever corridor was just used and lets its fatigue clock
# fall behind its thickness clock. Because heavy cold spells make far
# corridors' thickness grow fastest, this rotation also naturally shifts the
# biggest loads onto whichever corridor the temperature trend has made
# thickest that week, instead of being locked onto corridor 0 by habit.
import sys, json

MARGIN = 0.85
IDLE_BONUS_K = 0.02
IDLE_CAP_DAYS = 6

inst = json.load(sys.stdin)
n = inst["n_days"]
m = inst["mechanics"]
routes = inst["routes"]
temps = inst["temps"]
freeze = inst["freeze_point"]
nR = len(routes)

h = [rt["h0"] for rt in routes]
fatigue = [0.0] * nR
last_used = [-1000] * nR
routes_out = [-1] * n
masses_out = [0.0] * n

for d in range(n):
    best_score, best_r, best_mass = None, -1, 0.0
    for r in range(nR):
        rt = routes[r]
        eff = max(h[r] - fatigue[r], 0.0)
        ms = m["stress_limit"] * eff * eff / rt["length_factor"]
        margin_mass = MARGIN * ms
        if margin_mass <= 1e-6:
            continue
        idle = min(IDLE_CAP_DAYS, d - last_used[r]) * IDLE_BONUS_K * ms
        cand_score = margin_mass + idle
        if best_score is None or cand_score > best_score:
            best_score, best_r, best_mass = cand_score, r, margin_mass

    add = 0.0
    if best_r >= 0:
        routes_out[d] = best_r
        masses_out[d] = best_mass
        last_used[best_r] = d
        rt = routes[best_r]
        eff = max(h[best_r] - fatigue[best_r], 0.0)
        denom = m["stress_limit"] * max(eff, 1e-6) ** 2
        stress_frac = (best_mass * rt["length_factor"]) / denom
        add = m["fatigue_gain_k"] * stress_frac * h[best_r]

    for r in range(nR):
        if r == best_r:
            fatigue[r] = m["fatigue_decay"] * fatigue[r] + add
        else:
            fatigue[r] = m["fatigue_decay"] * fatigue[r]

    T = temps[d]
    for r in range(nR):
        rt = routes[r]
        g = rt["growth_rate"] * max(0.0, freeze - T) - rt["thaw_rate"] * max(0.0, T - freeze)
        h[r] = max(0.0, h[r] + g)

print(json.dumps({"routes": routes_out, "masses": masses_out}))
