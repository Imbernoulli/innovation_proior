# TIER: strong
# The insight: don't react to the visible error x[t] -- react to the PREDICTED state at the moment
# the command will actually land, x_hat[t+L]. Propagate the known model forward across the
# dead-time window using the public forecast for the disturbances in that window, and account for
# the effect of reserve commands already committed but not yet applied (the L-1 commands still
# "in flight"). This is a dead-time (Smith-predictor-style) compensator: it damps the deterministic
# delay itself instead of chasing the symptom currently on the dial.
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
        # predict the state at t+L: propagate x[t] forward L steps using known forecast
        # disturbances and the already-issued, not-yet-applied commands u[t-L+1 .. t-1]
        xhat = x
        for i in range(L):
            di = d[t + i] if t + i < T else 0.0
            src = t - L + i
            u_pipe = u[src] if 0 <= src < t else 0.0
            xhat = a * xhat + di - b * u_pipe
        ut = (0.9 * xhat) / b            # damp the predicted post-lag state
        ut = max(-u_max, min(u_max, ut))
        u.append(ut)
        u_eff = u[t - L] if t >= L else 0.0
        x = a * x + d[t] - b * u_eff

    print(json.dumps({"u": u}))


if __name__ == "__main__":
    main()
