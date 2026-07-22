# TIER: greedy
# The obvious first policy: reflexively spray full dose whenever the pest
# population is currently above the damage threshold, otherwise do nothing.
# Never releases predators (no reason to, in this mindset -- pesticide looks
# like the direct lever on the thing being measured). Since the plan must be
# committed in one shot, this rule is applied by internally re-simulating
# the SAME dynamics the evaluator uses (every constant needed is public) and
# reacting day-by-day inside that simulation -- exactly what a reactive
# controller would do if it could watch the season unfold. It never accounts
# for the fact that repeated spraying builds resistance (weakening every
# future dose against the pest) and keeps killing off the predator
# population that never gets to establish. On fields where the chemistry
# punishes repeated dosing this enters a costly, self-defeating cycle.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
P0, Q0 = inst["P0"], inst["Q0"]
r_pest, K_pest = inst["r_pest"], inst["K_pest"]
attack = inst["attack_rate"]
convert = inst["convert_eff"]
pred_death = inst["pred_death"]
resist_gain = inst["resist_gain"]
spray_eff0, pred_kill0 = inst["spray_eff0"], inst["pred_kill0"]
damage_thresh = inst["damage_thresh"]

spray = [0.0] * T
release = [0.0] * T

P, Q, R = P0, Q0, 0.02
for t in range(T):
    s = 1.0 if P > damage_thresh else 0.0
    spray[t] = s
    kill_pest = spray_eff0 * s * (1.0 - R)
    kill_pred = pred_kill0 * s
    P_next = max(0.0, P + r_pest * P * (1.0 - P / K_pest) - attack * P * Q - kill_pest * P)
    Q_next = max(0.0, Q * (1.0 - pred_death) + convert * attack * P * Q - kill_pred * Q)
    R_next = min(1.0, R + resist_gain * s * (1.0 - R))
    P, Q, R = P_next, Q_next, R_next

print(json.dumps({"spray": spray, "release": release}))
