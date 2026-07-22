#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0866 -- "Reserve Dispatch Against a Dead-Time Actuator"
(family: deadtime-damped-regulation; format B, quality-metric).

THEME.  A microgrid frequency deviation x[t] evolves under a known linear plant driven by a
load/generation imbalance d[t] and corrected by dispatched reserve u[t]:
    x[t+1] = a*x[t] + d[t] - b*u[t-L]      (u[s] := 0 for s < 0)
The reserve actuator has a DETERMINISTIC DEAD-TIME L: a command issued at step t only lands L
steps later (mechanism "actuator-deadtime-lag"). Cost sums quadratic tracking + effort, PLUS a
fixed penalty `pen` charged EVERY step |x[t]| exceeds a protection-relay threshold `theta` -- a
hard cliff modelling cascading instability once the grid trips (mechanism
"oscillation-instability-threshold"). The candidate PROGRAM is the dispatch planner itself: it
reads system parameters + a public day-ahead disturbance FORECAST and must output the full
T-step reserve command sequence in one shot. The instance's TRUE realized disturbance differs from
the forecast by a small bounded correction that is NEVER revealed to the candidate (kept in
inst["hidden"]) -- so a candidate cannot just solve an exact deterministic optimum against what it
is shown; it must be robust.

THE TRAP.  The obvious first design is a proportional/deadbeat controller reacting to the
CURRENTLY VISIBLE deviation x[t] as if the command took effect immediately. Because the true
dynamics apply u[t] only at t+L, by the time it lands the state has moved on and previously
issued, still-in-flight commands (t-L+1 .. t-1) are double-counted -- this compounds into growing
oscillation that repeatedly re-crosses theta whenever L is not small relative to the grid's memory
1/(1-a), or the disturbance changes abruptly (mechanism composition: the dead-time interacting
with the threshold cliff). THE INSIGHT (innovation hook): command reserves against the PREDICTED
state at the moment the command will actually land (t+L) -- propagate the known model forward over
the dead-time window using the forecast and the already-committed in-flight commands, and correct
THAT -- damping the deterministic delay itself instead of the currently visible error.

CANDIDATE CONTRACT (isolated stdin -> stdout program, called once per instance).
  stdin : ONE JSON object (the PUBLIC instance):
            {"T":int, "L":int, "a":float, "b":float, "c_eff":float, "theta":float, "pen":float,
             "u_max":float, "x0":float, "d_forecast":[T floats]}
  stdout: ONE JSON object:
            {"u": [T floats]}                      # |u[t]| <= u_max + 1e-6 required
  Any malformed / wrong-length / non-finite / out-of-range answer, a crash, a timeout, or non-JSON
  output -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  score(inst, answer) simulates the TRUE trajectory with the
hidden realized disturbance (forecast + a bounded, seeded correction never sent to the candidate),
computes cost = c_eff*sum(u^2) + sum_t [x[t]^2 + (pen if |x[t]|>theta else 0)], and normalizes
against baseline(inst) = cost of a constant feedforward-mean dispatch (evaluator's own trivial
construction, never a candidate file):
    r = clamp( 0.1 + 0.75 * (baseline - cost) / baseline, 0, 1 )
Matching the feedforward baseline scores 0.1; near-zero cost approaches ~0.85 (deliberate
headroom); doing WORSE than the feedforward baseline (e.g. by tripping the relay repeatedly)
scores 0.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via isorun.run_candidate; it
sees only inst["public"] (the forecast, never the hidden realized-disturbance correction).

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over the 10 instances, in [0,1]>
  Vector: [r_1, ..., r_10]
"""
import sys
import json
import math
import random
import hashlib
import isorun

TIMEOUT = 5
PEN_MULT = 3.0
NOISE_FRAC = 0.15
R_SLOPE = 0.75


# ============================== disturbance shapes (public forecast) =========================
def _make_forecast(kind, T, rng):
    d = [0.0] * T
    if kind == "step":
        h = rng.uniform(0.8, 1.2) * rng.choice([-1, 1])
        t0 = T // 4
        for t in range(t0, T):
            d[t] = h
    elif kind == "double_step":
        h1 = rng.uniform(0.7, 1.1) * rng.choice([-1, 1])
        h2 = rng.uniform(0.7, 1.1) * rng.choice([-1, 1])
        t0, t1 = T // 5, (3 * T) // 5
        for t in range(t0, t1):
            d[t] = h1
        for t in range(t1, T):
            d[t] = h1 + h2
    elif kind == "sine":
        amp = rng.uniform(0.5, 0.9)
        w = rng.uniform(0.15, 0.35)
        ph = rng.uniform(0, 2 * math.pi)
        for t in range(T):
            d[t] = amp * math.sin(w * t + ph)
    elif kind == "ramp":
        h = rng.uniform(0.02, 0.05) * rng.choice([-1, 1])
        cap = T * 0.6
        for t in range(T):
            d[t] = h * t if t < cap else h * cap
    elif kind == "noisy_small":
        for t in range(T):
            d[t] = rng.uniform(-0.3, 0.3)
    elif kind == "mixed":
        amp = rng.uniform(0.3, 0.6)
        w = rng.uniform(0.2, 0.4)
        ph = rng.uniform(0, 2 * math.pi)
        h = rng.uniform(0.5, 0.9) * rng.choice([-1, 1])
        t0 = T // 3
        for t in range(T):
            v = amp * math.sin(w * t + ph)
            if t >= t0:
                v += h
            v += rng.uniform(-0.15, 0.15)
            d[t] = v
    else:
        raise ValueError(kind)
    return d


# name, seed, T, L, a, b, c_eff, theta, u_max, kind
_SPECS = [
    ("mild-slow",           1,  70, 1, 0.90, 1.0, 0.08, 2.5, 2.0, "step"),
    ("mild-noisy",          2,  70, 2, 0.92, 1.0, 0.08, 2.5, 2.0, "noisy_small"),
    ("moderate-step",       3,  80, 3, 0.93, 1.0, 0.06, 2.5, 2.0, "step"),
    ("biglag-step",         4,  90, 6, 0.95, 1.0, 0.05, 2.5, 2.5, "step"),
    ("biglag-double-step",  5, 100, 7, 0.90, 1.0, 0.05, 2.5, 2.5, "double_step"),
    ("resonant-sine",       6, 100, 5, 0.92, 1.0, 0.05, 2.5, 2.5, "sine"),
    ("ramp-moderate",       7,  90, 4, 0.94, 1.0, 0.06, 2.5, 2.0, "ramp"),
    ("longlag-slow-decay",  8, 110, 8, 0.97, 1.0, 0.04, 2.5, 2.5, "step"),
    ("small-lag-bigdist",   9,  70, 1, 0.90, 1.0, 0.06, 3.5, 3.0, "step"),
    ("holdout-mixed",      10, 100, 6, 0.93, 0.9, 0.05, 2.5, 2.5, "mixed"),
]


def make_instances():
    insts = []
    for name, seed, T, L, a, b, c_eff, theta, u_max, kind in _SPECS:
        rng = random.Random(seed)
        d_forecast = _make_forecast(kind, T, rng)
        amp = max((abs(v) for v in d_forecast), default=1.0) or 1.0
        # Hidden realized-disturbance correction: seeded from a SHA-256 digest of the instance
        # name plus a fixed internal salt (not a small guessable integer derived from public
        # fields) so the exact noise draw cannot be reconstructed from the public instance JSON
        # alone. The candidate never receives this seed or `hidden` at all (isorun sends only
        # `public`); this hardening only removes the theoretical closed-form-reconstruction path.
        digest = hashlib.sha256(f"fsx_A_0866|hidden-noise|{name}|v1".encode()).hexdigest()
        rng2 = random.Random(int(digest[:16], 16))
        eps = [rng2.uniform(-NOISE_FRAC * amp, NOISE_FRAC * amp) for _ in range(T)]
        d_actual = [f + e for f, e in zip(d_forecast, eps)]
        pen = PEN_MULT * theta * theta
        public = {"T": T, "L": L, "a": a, "b": b, "c_eff": c_eff, "theta": theta,
                  "pen": pen, "u_max": u_max, "x0": 0.0, "d_forecast": list(d_forecast)}
        insts.append({"public": public, "hidden": {"d_actual": d_actual}, "name": name})
    return insts


# ============================== dynamics / cost ===============================================
def _simulate(inst, u_seq, d):
    p = inst["public"]
    T, L, a, b, x0 = p["T"], p["L"], p["a"], p["b"], p["x0"]
    x = [0.0] * (T + 1)
    x[0] = x0
    for t in range(T):
        u_eff = u_seq[t - L] if t >= L else 0.0
        x[t + 1] = a * x[t] + d[t] - b * u_eff
    return x


def _cost(inst, u_seq, x):
    p = inst["public"]
    T, c_eff, theta, pen = p["T"], p["c_eff"], p["theta"], p["pen"]
    total = c_eff * sum(u * u for u in u_seq)
    for t in range(1, T + 1):
        total += x[t] * x[t]
        if abs(x[t]) > theta:
            total += pen
    return total


def baseline(inst):
    p = inst["public"]
    T = p["T"]
    d_actual = inst["hidden"]["d_actual"]
    m = sum(p["d_forecast"]) / T
    u_seq = [m] * T
    x = _simulate(inst, u_seq, d_actual)
    return _cost(inst, u_seq, x)


# ============================== answer validation ==============================================
def _validate_u(answer, T, u_max):
    if not isinstance(answer, dict):
        return None
    u = answer.get("u")
    if not isinstance(u, list) or len(u) != T:
        return None
    out = []
    tol = 1e-6
    for v in u:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        try:
            fv = float(v)
        except (OverflowError, ValueError):
            return None
        if fv != fv or fv in (float("inf"), float("-inf")):
            return None
        if fv < -u_max - tol or fv > u_max + tol:
            return None
        out.append(fv)
    return out


def score(inst, answer):
    p = inst["public"]
    u_seq = _validate_u(answer, p["T"], p["u_max"])
    if u_seq is None:
        return False, None
    d_actual = inst["hidden"]["d_actual"]
    x = _simulate(inst, u_seq, d_actual)
    obj = _cost(inst, u_seq, x)
    if not (obj == obj) or obj in (float("inf"), float("-inf")):
        return False, None
    return True, obj


# ============================== driver ==========================================================
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    insts = make_instances()

    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=TIMEOUT)
        if st != "OK":
            vec.append(0.0)
            continue
        ok, obj = score(inst, ans)
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        r = 0.1 + R_SLOPE * (b - obj) / max(b, 1e-12)
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
