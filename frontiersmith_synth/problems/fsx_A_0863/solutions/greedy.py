# TIER: greedy
# The obvious recipe an average strong coder writes first: a PROPORTIONAL
# controller with ONE FIXED gain, chosen from the only global scale the
# public instance advertises (the full pH axis mapped onto the full
# titrant budget): Kp = V_max / 14.0. Step = Kp * (target - pH). This DOES
# react to the live pH reading (unlike `trivial`), but the gain is never
# re-tuned to the LOCAL buffer capacity: on a gentle, high-capacity
# plateau the true slope is far below the average the gain assumes (so it
# creeps and burns rounds), and once it reaches a steep, low-capacity
# equivalence jump the true slope is far ABOVE the average (so the very
# same "reasonable-looking" correction rockets straight past the target).
import sys, json

inst = json.load(sys.stdin)
V_max = float(inst["V_max"])
pH = float(inst["pH"])
target = float(inst["target_pH"])
V = float(inst["V"])
max_add = float(inst.get("max_add", V_max))

if pH >= target - 1e-9:
    add = 0.0
else:
    Kp = V_max / 14.0
    remaining = V_max - V
    cap = min(remaining, max_add)
    add = Kp * (target - pH)
    add = max(0.0, min(add, cap))
    if add <= 1e-9:
        add = min(cap, cap * 0.05) if cap > 0 else 0.0

print(json.dumps({"add": add}))
