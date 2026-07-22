# TIER: strong
# The insight: the ecosystem has a self-regulating fixed point. At the
# critical predator density q* = r_pest/attack_rate, predation alone
# (attack_rate*P*q*) cancels the pest's logistic growth at low P, so once Q
# reaches q* the pest is suppressed FOR FREE and no further chemical is
# needed. The policy therefore rations chemical use and spends its budget on
# ESTABLISHING and preserving the predator population instead of reacting to
# the pest directly:
#   1. Read spray_eff0, pred_kill0, resist_gain to judge whether this
#      field's chemistry makes a short, calibrated opening spray worth its
#      resistance/predator cost (chem_index = spray_eff0/(pred_kill0 +
#      resist_gain); high => cheap, safe chemical control -- take a brief
#      2-day knockdown before predators can establish; low => any spray is
#      self-defeating -- skip it and eat the transient loss while predators
#      ramp up).
#   2. Every day Q is below q*, release predators up to release_cap; once Q
#      reaches q*, stop paying release cost (maintenance is automatic via
#      the conversion term).
#   3. Keep a rare last-resort valve: only spray again if the pest is truly
#      running away toward carrying capacity AND resistance is still low
#      enough for it to matter.
# This plan is committed once, but it is built by internally re-simulating
# the SAME public dynamics the evaluator will replay.
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
release_cap = inst["release_cap"]

q_star = r_pest / attack
chem_index = spray_eff0 / (pred_kill0 + resist_gain)
allow_bootstrap = chem_index > 1.2

spray = [0.0] * T
release = [0.0] * T

P, Q, R = P0, Q0, 0.02
bootstrap_days_left = 2 if allow_bootstrap and P0 > damage_thresh else 0
for t in range(T):
    s = 0.0
    u = 0.0

    if bootstrap_days_left > 0 and P > damage_thresh:
        s = 1.0
        bootstrap_days_left -= 1

    if Q < q_star:
        u = release_cap

    # last-resort emergency valve: pest genuinely running away toward
    # carrying capacity while resistance is still low enough to bite
    if P > 0.85 * K_pest and R < 0.6:
        s = max(s, 0.6)

    spray[t] = s
    release[t] = u

    kill_pest = spray_eff0 * s * (1.0 - R)
    kill_pred = pred_kill0 * s
    P_next = max(0.0, P + r_pest * P * (1.0 - P / K_pest) - attack * P * Q - kill_pest * P)
    Q_next = max(0.0, Q * (1.0 - pred_death) + convert * attack * P * Q - kill_pred * Q + u)
    R_next = min(1.0, R + resist_gain * s * (1.0 - R))
    P, Q, R = P_next, Q_next, R_next

print(json.dumps({"spray": spray, "release": release}))
