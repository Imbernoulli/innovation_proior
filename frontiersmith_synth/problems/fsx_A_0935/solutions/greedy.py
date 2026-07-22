# TIER: greedy
import sys, json, math

inst = json.load(sys.stdin)
m, c, k = float(inst["m"]), float(inst["c"]), float(inst["k"])

# "obvious" aggressive pole-placement: push the closed-loop natural frequency to 4x the plant's
# own open-loop frequency for a fast response, tuned purely from the plant model -- never looks
# at the published disturbance spectrum at all. Damping is left light (classic "chase speed"
# mistake) which leaves a real resonance peak sitting wherever the closed loop happens to land.
w0 = math.sqrt(k / m)
w_t = 4.0 * w0
zeta_t = 0.18

kp = max(m * w_t * w_t - k, 0.0)
kd = max(2.0 * zeta_t * w_t * m - c, 0.0)
ki = 0.4 * kp / w_t if w_t > 1e-9 else 0.0

print(json.dumps({"kp": kp, "kd": kd, "ki": ki, "resonators": []}))
