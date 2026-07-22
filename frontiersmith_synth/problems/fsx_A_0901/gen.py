#!/usr/bin/env python3
"""
gen.py <testId>  ->  prints ONE noisy TRAIN trace to stdout.

Lazy-throttle autopilot recovery.  A hidden PI altitude-hold law

    e[t]  = sp[t] - y[t]                     (tracking error)
    I[t]  = I[t-1] + e[t]      (I[-1]=0)     (accumulated correction)
    u[t]  = Umax * tanh( (Kp*e[t] + Ki*I[t]) / Umax )   (lazy-throttle law)

drives a KNOWN first-order plant  y[t+1] = y[t] + alpha*(u[t]-y[t]).  Each
testId fixes a DIFFERENT hidden (Kp, Ki, Umax, alpha).

The solver only ever SEES this TRAIN trace, logged on a CALM flight: small
setpoint corrections that never push the throttle anywhere near saturation.
In that quasi-linear regime u looks almost exactly like a plain linear
combination Kp*e + Ki*I -- the tanh compression only shows up as a faint,
barely-visible curvature in the residuals.  The held-out grading flight is a
STORMY profile with LARGE course changes (regenerated only inside the
grader) where the throttle's limited authority, and the windup it causes,
fully dominate the response.

STDOUT prints ONLY: a header "<n_rows> <test_id> <alpha>" then n_rows rows
"<sp> <y_noisy> <u_noisy>".  The hidden Kp, Ki, Umax and the RNG seeds are
NOT printed anywhere.
"""
import sys, random


def hidden_law(t):
    """Hidden lazy-throttle law for this test id (lives in gen AND grader, never printed)."""
    rng = random.Random(900001 + t * 7919)
    Kp = rng.uniform(0.6, 1.4)
    Ki = rng.uniform(0.05, 0.25)
    Umax = rng.uniform(3.0, 5.5)
    alpha = rng.uniform(0.15, 0.35)
    return Kp, Ki, Umax, alpha


def tanh(x):
    if x > 40.0:
        return 1.0
    if x < -40.0:
        return -1.0
    e2 = pow(2.718281828459045, 2.0 * x)
    return (e2 - 1.0) / (e2 + 1.0)


def calm_setpoints(t, n):
    """SLOW, small-amplitude training setpoint (never saturates the throttle)."""
    rng = random.Random(31337 + t * 104729)
    sp = []
    cur = 0.0
    steps_left = 0
    for _ in range(n):
        if steps_left <= 0:
            cur = rng.uniform(-1.6, 1.6)
            steps_left = rng.randint(15, 30)
        sp.append(cur)
        steps_left -= 1
    return sp


def true_rollout(sp, Kp, Ki, Umax, alpha):
    n = len(sp)
    y = [0.0] * (n + 1)
    I = 0.0
    u = []
    for t in range(n):
        e = sp[t] - y[t]
        I = I + e
        ut = Umax * tanh((Kp * e + Ki * I) / Umax)
        y[t + 1] = y[t] + alpha * (ut - y[t])
        u.append(ut)
    return y[:n], u


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    Kp, Ki, Umax, alpha = hidden_law(t)

    n = 260 - 6 * (t - 1)  # difficulty ladder: less data as testId grows
    sigma = 0.024

    sp = calm_setpoints(t, n)
    y_true, u_true = true_rollout(sp, Kp, Ki, Umax, alpha)

    rng = random.Random(555 + t * 13)
    y_obs = [yv + rng.gauss(0.0, sigma) for yv in y_true]
    u_obs = [uv + rng.gauss(0.0, sigma) for uv in u_true]

    out = ["%d %d %.6f" % (n, t, alpha)]
    for i in range(n):
        out.append("%.6f %.6f %.6f" % (sp[i], y_obs[i], u_obs[i]))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
