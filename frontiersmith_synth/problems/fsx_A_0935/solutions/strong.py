# TIER: strong
import sys, json

inst = json.load(sys.stdin)

# Push the base loop to the validator's own gain/damping BUDGET LIMIT (kp<=320, kd<=13) --
# whatever a spectrum-ignorant search would also do, since more base gain is close to a free
# win for a linear plant up to that budget. That alone is NOT enough (a search confirms the
# best pure-PID design at this same budget still falls well short). The decisive move on top:
# read the PUBLISHED (freq, amplitude) list and drop one internal-model resonator on each of
# the highest-amplitude published sinusoids (up to the allowed budget) -- the sweep itself is
# treated as the design specification, and this is what actually closes the remaining gap that
# no amount of extra base gain (there isn't any left to give) can reach.
kp = 320.0
kd = 13.0
ki = 8.0

max_res = int(inst.get("max_resonators", 6))
sins = sorted(inst.get("sinusoids", []), key=lambda s: -float(s["amp"]))[:max_res]
resonators = [{"freq": float(s["freq"]), "zeta": 0.03, "gain": 26.0} for s in sins]

print(json.dumps({"kp": kp, "kd": kd, "ki": ki, "resonators": resonators}))
