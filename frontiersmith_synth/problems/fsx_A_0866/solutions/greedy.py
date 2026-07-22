# TIER: greedy
# The obvious first design: a deadbeat proportional controller that cancels the CURRENTLY VISIBLE
# deviation, using the true a/b/d model but WITHOUT accounting for the actuator's L-step dead-time
# (it computes the command as if it lands next step, when it really lands L steps later). This is
# the classic textbook "cancel the error you can see" recipe -- it never reserves against the
# predicted post-lag state, so already-in-flight commands get double-counted and, when L is not
# small relative to the grid's memory or the disturbance changes abruptly, it overshoots into
# sustained oscillation across the instability threshold.
import sys
import json


def main():
    inst = json.load(sys.stdin)
    T = inst["T"]
    L = inst["L"]
    a = inst["a"]
    b = inst["b"]
    d = inst["d_forecast"]
    u_max = inst["u_max"]
    x0 = inst["x0"]

    x = x0
    u = []
    for t in range(T):
        ut = (a * x + d[t]) / b          # cancel CURRENT visible x, ignoring the L-step lag
        ut = max(-u_max, min(u_max, ut))
        u.append(ut)
        u_eff = u[t - L] if t >= L else 0.0
        x = a * x + d[t] - b * u_eff

    print(json.dumps({"u": u}))


if __name__ == "__main__":
    main()
