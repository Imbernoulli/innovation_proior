#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0935 -- "One Controller Versus Twenty-Five Hostile Disturbances"
(family: notch-shaping-disturbance-controller; format B, quality-metric).

THEME. A single mass-spring-damper plant (state x = position, v = velocity)
    m*x'' + c*x' + k*x = u(t) + d(t)
must be held at x=0 by a feedback controller that outputs force u(t), while the evaluator
replays the SAME plant, closed under the SAME controller, against a PUBLISHED suite of 25
disturbance profiles d(t): 15 sinusoids at listed (freq, amplitude, phase), 6 transient pulse
trains, and 4 slow bias/ramp drifts (mechanism "disturbance-frequency-sweep" -- the whole
spectrum, not just an aggregate, is handed to the candidate in the public instance). The plant is
integrated with a fixed-step RK4 rollout (mechanism "deterministic-plant-rollout"); score is the
worst-case (plus a small mean term) integrated |tracking error| over the 25 profiles, subject to a
smooth actuation ceiling (tanh saturation at u_max).

CANDIDATE CONTRACT (isolated stdin -> stdout program, called once per instance). The candidate is
the controller DESIGNER: it reads the plant + full published disturbance suite and outputs a
closed-form feedback LAW (mechanism "feedback-program-synthesis") -- not raw force samples:
  stdin : ONE JSON object (the PUBLIC instance):
    {"m":float,"c":float,"k":float,"u_max":float,"dt":float,"T":float,
     "sinusoids":[{"freq":f,"amp":a,"phase":p}, ...15],
     "impulses":[[[t0,mag], ...], ...6],       # each profile = list of (time, magnitude) pulses
     "drifts":[{"bias":b,"ramp":r}, ...4],
     "max_resonators": 6}
  stdout: ONE JSON object:
    {"kp":float, "kd":float, "ki":float,
     "resonators":[{"freq":float,"zeta":float,"gain":float}, ... up to max_resonators]}
The evaluator builds the closed-loop force law from these parameters itself and simulates it; the
candidate program never touches the simulator.

  u_raw(t) = kp*e(t) + kd*(-v(t)) + ki*I(t) - sum_k gain_k * r_k(t),   e = -x
  u(t)     = u_max * tanh(u_raw(t) / u_max)                            [smooth actuator ceiling]
Each resonator k is itself a small stable 2nd-order filter driven by the error signal:
  r_k'' + 2*zeta_k*w_k*r_k' + w_k^2*r_k = w_k^2 * e(t)
which -- when w_k is set to EXACTLY a published disturbance frequency and zeta_k is small -- acts
as an internal-model absorber that nearly cancels the closed-loop's steady response to that one
listed sinusoid (classic internal-model-principle loop shaping), at some actuation cost.

kp/kd are deliberately capped tight (kp<=320, kd<=13; see KP_MAX/KD_MAX) relative to what these
plants would need to push the closed loop's own bandwidth and damping high enough to flatten out
resonance everywhere by brute force. A grid search over EVERY (kp,kd,ki) with no resonators,
including simply maxing both gains out at the cap, tops out at ~0.71 mean ratio; the resonator-
augmented reference reaches ~0.78 -- there is no leftover gain budget a spectrum-ignorant search
could spend to close that gap, only the resonators can.

THE TRAP. The obvious first design (the "greedy" tier) tunes a single aggressive PD pole
placement from the plant's own open-loop frequency w0=sqrt(k/m) -- push the closed-loop natural
frequency to 4*w0 for a "fast" response -- but with LIGHT damping (zeta=0.18, the classic
Ziegler-Nichols-flavoured mistake of chasing speed over damping). On >=3 of the 10 instances the
generator plants one published sinusoid at EXACTLY that resulting closed-loop frequency with
10x-amplified magnitude; the underdamped greedy loop resonates there and that one profile
dominates the worst-case aggregate. THE INSIGHT: within the tight gain budget, read the
published (freq, amplitude) list and drop internal-model resonators exactly on the
highest-amplitude published frequencies -- the sweep itself IS the design specification, not a
scenario to tune against on average, and no in-budget base-gain retune reaches the same result
(the best gain-only search under the same caps tops out well below what resonators unlock).

SCORING. score(inst, answer) validates and clips nothing silently (out-of-range/non-finite ->
reject); simulates all 25 profiles; obj = max_j(E_j) + 0.25*mean_j(E_j) + 0.05*mean_j(energy_j)
(E_j = dt-integrated |x(t)| for profile j; energy_j = dt-integrated (u(t)/u_max)^2, a mild
actuation-effort cost). Normalized against baseline(inst) = a fixed, plant-naive weak PD (no
resonators, no plant-specific tuning) the evaluator computes itself:
    r = clamp(0.1 + 0.75*(baseline - obj)/max(baseline,eps), 0, 1)
Matching the weak baseline scores ~0.1; beating it substantially approaches (but is capped well
below) 1.0, leaving headroom above the "strong" reference.

ISOLATION. The candidate is untrusted and runs in a FRESH SUBPROCESS via isorun.run_candidate; it
never sees evaluator internals, only inst["public"].

CLI: python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over the 10 instances, in [0,1]>
  Vector: [r_1, ..., r_10]
"""
import sys
import json
import math
import random
import isorun

TIMEOUT = 20
N_SIN = 15
N_IMP = 6
N_DRIFT = 4
N_MAX_RES = 6
DT = 0.02
T_HORIZON = 5.0
N_STEPS = int(round(T_HORIZON / DT))
R_SLOPE = 0.75
MEAN_WEIGHT = 0.25
ENERGY_WEIGHT = 0.05
PULSE_SIGMA = 0.10
KP_MAX = 320.0
KD_MAX = 13.0

# trap instances: >=3 of the 10 get one published sinusoid planted exactly at greedy's own
# resulting closed-loop resonance frequency, at amplified amplitude.
TRAP_INSTANCES = {0, 3, 5, 8}

# "obvious" greedy pole-placement constants (must match solutions/greedy.py exactly, since the
# generator plants the resonance collision against THIS specific formula)
GREEDY_W_MULT = 4.0
GREEDY_ZETA = 0.18

# weak, plant-naive baseline controller used ONLY for score normalization (never a candidate file)
BASE_KP = 1.5
BASE_KD = 0.5
BASE_KI = 0.0


# ============================== instance generation =============================================
# name, seed, m-range, k-range, zeta_open-range
_SPECS = [
    ("light-soft",     101, (0.9, 1.3), (4.0, 6.5),  (0.02, 0.05)),
    ("light-stiff",    102, (0.8, 1.2), (14.0, 20.0), (0.02, 0.05)),
    ("mid-soft",       103, (1.2, 1.8), (5.0, 8.0),  (0.03, 0.06)),
    ("mid-mid",        104, (1.3, 2.0), (8.0, 12.0), (0.03, 0.07)),
    ("mid-stiff",      105, (1.0, 1.6), (12.0, 18.0), (0.02, 0.06)),
    ("heavy-soft",     106, (2.0, 2.5), (4.5, 7.0),  (0.04, 0.08)),
    ("heavy-mid",      107, (1.8, 2.4), (9.0, 13.0), (0.03, 0.07)),
    ("heavy-stiff",    108, (2.0, 2.5), (15.0, 20.0), (0.02, 0.05)),
    ("holdout-soft",   109, (0.9, 1.4), (4.0, 7.0),  (0.03, 0.08)),
    ("holdout-stiff",  110, (1.5, 2.2), (13.0, 20.0), (0.02, 0.06)),
]


def make_instances():
    insts = []
    for idx, (name, seed, mrng, krng, zrng) in enumerate(_SPECS):
        rng = random.Random(seed)
        m = round(rng.uniform(*mrng), 6)
        k_plant = round(rng.uniform(*krng), 6)
        zeta_open = rng.uniform(*zrng)
        c_plant = round(2.0 * zeta_open * math.sqrt(k_plant * m), 6)
        # w0/w_trap computed from the ROUNDED (published) m, k, c so the trap collision is
        # bit-exact with what a candidate recomputes from the public instance it receives.
        w0 = math.sqrt(k_plant / m)

        is_trap = idx in TRAP_INSTANCES
        w_trap = GREEDY_W_MULT * w0

        base_amp = 0.10 * k_plant

        sinusoids = []
        if is_trap:
            sinusoids.append({"freq": w_trap, "amp": 10.0 * base_amp, "phase": rng.uniform(0, 2 * math.pi)})
            n_rest = N_SIN - 1
        else:
            n_rest = N_SIN
        for _ in range(n_rest):
            # spread across a band around w0 avoiding a near-exact re-collision with w_trap
            while True:
                w = rng.uniform(0.35 * w0, 6.0 * w0)
                if not is_trap or abs(w - w_trap) > 0.12 * w_trap:
                    break
            amp = rng.uniform(0.55, 1.5) * base_amp
            sinusoids.append({"freq": w, "amp": amp, "phase": rng.uniform(0, 2 * math.pi)})
        rng.shuffle(sinusoids)

        impulses = []
        for _ in range(N_IMP):
            n_pulses = rng.choice([1, 1, 2])
            pulses = []
            for _ in range(n_pulses):
                t0 = rng.uniform(0.3, T_HORIZON - 0.3)
                mag = rng.uniform(0.6, 1.4) * base_amp * rng.choice([-1.0, 1.0]) * 3.0
                pulses.append([round(t0, 4), round(mag, 6)])
            impulses.append(pulses)

        drifts = []
        for _ in range(N_DRIFT):
            bias = rng.uniform(0.3, 0.9) * base_amp * rng.choice([-1.0, 1.0])
            ramp = rng.uniform(0.0, 0.35) * base_amp * rng.choice([-1.0, 1.0])
            drifts.append({"bias": round(bias, 6), "ramp": round(ramp, 6)})

        max_sin_amp = max(s["amp"] for s in sinusoids)
        max_imp_peak = max(abs(mag) for pulses in impulses for _, mag in pulses)
        max_drift_steady = max(abs(dr["bias"]) + abs(dr["ramp"]) * T_HORIZON for dr in drifts)
        u_max = 1.5 * max(max_sin_amp, max_imp_peak, max_drift_steady)

        public = {
            "m": round(m, 6), "c": round(c_plant, 6), "k": round(k_plant, 6),
            "u_max": round(u_max, 6),
            "dt": DT, "T": T_HORIZON,
            "sinusoids": [{"freq": round(s["freq"], 6), "amp": round(s["amp"], 6),
                           "phase": round(s["phase"], 6)} for s in sinusoids],
            "impulses": impulses,
            "drifts": drifts,
            "max_resonators": N_MAX_RES,
        }
        insts.append({"public": public, "hidden": {}, "name": name})
    return insts


# ============================== dynamics ==========================================================
def _pulse_train(pulses, t):
    v = 0.0
    for t0, mag in pulses:
        dt = t - t0
        v += mag * math.exp(-0.5 * (dt / PULSE_SIGMA) ** 2)
    return v


def _dist_value(profile, t):
    kind = profile[0]
    if kind == "sin":
        _, freq, amp, phase = profile
        return amp * math.sin(freq * t + phase)
    if kind == "imp":
        _, pulses = profile
        return _pulse_train(pulses, t)
    if kind == "drift":
        _, bias, ramp = profile
        return bias + ramp * t
    raise ValueError(kind)


def _build_profiles(public):
    profs = []
    for s in public["sinusoids"]:
        profs.append(("sin", s["freq"], s["amp"], s["phase"]))
    for pulses in public["impulses"]:
        profs.append(("imp", [tuple(p) for p in pulses]))
    for dr in public["drifts"]:
        profs.append(("drift", dr["bias"], dr["ramp"]))
    return profs


def _compute_u(y, u_max, ctrl):
    n_res = len(ctrl["resonators"])
    x, v, I = y[0], y[1], y[2]
    e = -x
    u_raw = ctrl["kp"] * e + ctrl["kd"] * (-v) + ctrl["ki"] * I
    for j in range(n_res):
        u_raw -= ctrl["resonators"][j]["gain"] * y[3 + 2 * j]
    u = u_max * math.tanh(u_raw / u_max) if u_max > 1e-9 else 0.0
    return u_raw, u


def _deriv(y, t, m, c, k, u_max, ctrl, profile):
    n_res = (len(y) - 3) // 2
    x, v, I = y[0], y[1], y[2]
    e = -x
    u_raw, u = _compute_u(y, u_max, ctrl)
    dy = [0.0] * len(y)
    d = _dist_value(profile, t)
    dy[0] = v
    dy[1] = (-k * x - c * v + u + d) / m
    # simple anti-windup clamp on the integrator state
    I_MAX = 50.0
    if abs(I) >= I_MAX and (I > 0) == (e > 0):
        dy[2] = 0.0
    else:
        dy[2] = e
    for j in range(n_res):
        r = y[3 + 2 * j]
        rd = y[3 + 2 * j + 1]
        w = ctrl["resonators"][j]["freq"]
        z = ctrl["resonators"][j]["zeta"]
        dy[3 + 2 * j] = rd
        dy[3 + 2 * j + 1] = -2.0 * z * w * rd - w * w * r + w * w * e
    return dy


def _rk4_step(y, t, dt, m, c, k, u_max, ctrl, profile):
    def add(a, b, s):
        return [ai + s * bi for ai, bi in zip(a, b)]
    k1 = _deriv(y, t, m, c, k, u_max, ctrl, profile)
    k2 = _deriv(add(y, k1, dt / 2), t + dt / 2, m, c, k, u_max, ctrl, profile)
    k3 = _deriv(add(y, k2, dt / 2), t + dt / 2, m, c, k, u_max, ctrl, profile)
    k4 = _deriv(add(y, k3, dt), t + dt, m, c, k, u_max, ctrl, profile)
    return [y[i] + (dt / 6.0) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(len(y))]


def _rollout_error(public, ctrl, profile):
    m, c, k, u_max = public["m"], public["c"], public["k"], public["u_max"]
    n_res = len(ctrl["resonators"])
    y = [0.0] * (3 + 2 * n_res)
    t = 0.0
    dt = public["dt"]
    err = 0.0
    energy = 0.0
    for step in range(N_STEPS):
        err += abs(y[0]) * dt
        _, u = _compute_u(y, u_max, ctrl)
        energy += (u / u_max) ** 2 * dt if u_max > 1e-9 else 0.0
        y = _rk4_step(y, t, dt, m, c, k, u_max, ctrl, profile)
        t += dt
        if not all(v == v and abs(v) != float("inf") for v in y):
            return 1e6, 1e6
    return err, energy


def _aggregate(public, ctrl):
    profs = _build_profiles(public)
    errs, energies = zip(*(_rollout_error(public, ctrl, p) for p in profs))
    return (max(errs) + MEAN_WEIGHT * (sum(errs) / len(errs))
            + ENERGY_WEIGHT * (sum(energies) / len(energies)))


# ============================== answer validation =================================================
def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def _validate(answer, max_res):
    if not isinstance(answer, dict):
        return None
    kp, kd, ki = answer.get("kp"), answer.get("kd"), answer.get("ki")
    for v in (kp, kd, ki):
        if not _finite(v):
            return None
    kp, kd, ki = float(kp), float(kd), float(ki)
    if not (0.0 <= kp <= KP_MAX and 0.0 <= kd <= KD_MAX and 0.0 <= ki <= 200.0):
        return None
    res_in = answer.get("resonators", [])
    if not isinstance(res_in, list) or len(res_in) > max_res:
        return None
    resonators = []
    for r in res_in:
        if not isinstance(r, dict):
            return None
        freq, zeta, gain = r.get("freq"), r.get("zeta"), r.get("gain")
        if not (_finite(freq) and _finite(zeta) and _finite(gain)):
            return None
        freq, zeta, gain = float(freq), float(zeta), float(gain)
        if not (0.0 < freq <= 60.0 and 0.005 <= zeta <= 1.5 and abs(gain) <= 300.0):
            return None
        resonators.append({"freq": freq, "zeta": zeta, "gain": gain})
    return {"kp": kp, "kd": kd, "ki": ki, "resonators": resonators}


def score(inst, answer):
    ctrl = _validate(answer, inst["public"]["max_resonators"])
    if ctrl is None:
        return False, None
    obj = _aggregate(inst["public"], ctrl)
    if not (obj == obj) or abs(obj) == float("inf"):
        return False, None
    return True, obj


def baseline(inst):
    ctrl = {"kp": BASE_KP, "kd": BASE_KD, "ki": BASE_KI, "resonators": []}
    return _aggregate(inst["public"], ctrl)


# ============================== driver =============================================================
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
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
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
