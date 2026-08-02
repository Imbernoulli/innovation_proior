#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_1299 -- "Fuel is the Only Thing You Cannot Make More Of:
GEO Longitude Station-Keeping" (family: spacecraft-station-keep; format B, quality-metric).

THEME.  A geostationary satellite is assigned a longitude slot.  Earth's triaxiality
(the equatorial ellipse is not a perfect circle) pulls every GEO satellite toward one
of two stable longitudes with an almost-constant along-track acceleration whose SIGN
depends only on which side of the nearest stable point the slot sits on -- this is a
SYSTEMATIC drift, not noise.  On top of it ride small, effectively random
perturbations (solar-radiation-pressure fluctuations, mismodeled forces).  Both act on
the same scalar "position" (longitude offset from slot center, arbitrary units):

    x[t+1] = x[t] + bias + noise[t]        (noise[t] drawn uniformly in [-noise_amp, noise_amp])

The satellite must stay inside a station-keeping box |x| <= box_half_width.  A ground
controller may fire a correction burn at any step, instantly resetting x to a chosen
target_pos; a burn's fuel cost is proportional to the distance moved and INVERSELY
proportional to that step's burn efficiency e(t): cost = |x - target_pos| / e(t).
Efficiency is periodic -- e(t) = eff_high inside a short window (t mod period) <
window_len, and eff_low the rest of the cycle (real EW station-keeping burns are far
cheaper when timed to specific points in the orbit / ground-station geometry).  A
mission-control safety system forces an immediate burn (regardless of efficiency, using
target_pos) the instant |x| would cross 0.95 * box_half_width, so a slow policy cannot
be blindsided into an unrecoverable exit -- but that forced burn is usually expensive.

The satellite "dies" (mission ends) the first step |x| exceeds box_half_width.  The
objective is to MAXIMIZE mission lifetime (steps survived, capped at horizon) subject to
a finite fuel_budget.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
    {"name": str, "horizon": T (int), "box_half_width": W (float),
     "bias": b (float, can be + or -), "noise_amp": A (float, >=0),
     "fuel_budget": F (float), "period": P (int), "window_len": L (int, < P),
     "eff_high": float, "eff_low": float}
  stdout: ONE JSON object -- a CONTROL POLICY (not a per-step trajectory):
    {"band_lo": float >= 0,   # trigger a correction once x < -band_lo
     "band_hi": float >= 0,   # trigger a correction once x >  band_hi
     "target_pos": float,     # position a correction burn resets x to
     "patience": int >= 0}    # steps willing to wait, once triggered, for an
                               # eff_high window before burning anyway (unless the
                               # 0.95*W safety margin forces it sooner)

  Validity: all four fields present, correct type (band_lo/band_hi/target_pos numeric
  and finite; patience numeric, finite, non-negative -- rounded to an int), band_lo>=0,
  band_hi>=0, |target_pos| <= 1e6, 0 <= patience <= 10**6.  Anything else (missing
  field, wrong type, NaN/Inf, negative band/patience, crash, timeout, non-JSON) ->
  that instance scores 0.0.

SIMULATION (run by the evaluator, once per instance, using the FULL instance -- the
candidate never sees noise[t] or the RNG seed).  Each step: advance x by bias+noise[t];
if |x| > W the mission has already ended (lifetime = steps survived before this step);
otherwise check the dead-band, and if outside it and (currently in an eff_high window,
OR patience steps have already been spent waiting, OR the 0.95*W safety margin is
breached) fire a burn to target_pos at the step's efficiency, spending fuel (a burn
that would exceed remaining fuel is scaled down to whatever fuel remains).  Continue to
`horizon`; a satellite that never leaves the box scores lifetime = horizon.

SCORING (deterministic; no wall-time).  Per instance:
    base_life = lifetime achieved by the "do nothing" policy (band_lo=band_hi=10*W,
                so it never triggers) -- the evaluator computes this itself.
    cand_life = lifetime achieved by the candidate's policy.
    ub_life   = horizon (loose, generally-unreachable-under-a-tight-budget upper bound).
    r = clamp(0.1 + 0.9 * (cand_life - base_life) / max(1e-9, ub_life - base_life), 0, 1)
Reproducing "do nothing" scores ~0.1; surviving longer scores higher, capped at 1.0.
Final score is the mean of r over all instances (varied bias direction/magnitude, noise
scale, and efficiency-window width, including harder held-out profiles).

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance below.  The noise
sequence, RNG seed, and all scoring happen in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = [seed & ((1 << 64) - 1)]

    def uniform(lo, hi):
        state[0] = (state[0] * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        frac = (state[0] >> 11) / float(1 << 53)
        return lo + frac * (hi - lo)

    return uniform


# ----------------------------- instance family -----------------------------
# (seed, name, horizon T, box_half_width W, bias b, noise_amp A,
#  fuel_budget F, period P, window_len L)
_SPECS = [
    (1001, "slot-EastDrift-A",   391, 100.0,  0.55, 0.35,  88.21, 24, 4),
    (1002, "slot-WestDrift-A",   393, 100.0, -0.55, 0.35,  87.89, 24, 4),
    (1003, "slot-EastDrift-B",   549, 100.0,  0.40, 0.50,  92.86, 20, 4),
    (1004, "slot-WestDrift-B",   556, 100.0, -0.40, 0.50,  86.19, 20, 4),
    (1005, "slot-EastDrift-C",   789, 100.0,  0.30, 0.35,  91.09, 18, 3),
    (1006, "slot-EastDrift-D",   342, 100.0,  0.65, 0.25,  88.18, 30, 5),
    (1007, "slot-WestDrift-D",   336, 100.0, -0.65, 0.25,  88.02, 30, 5),
    (1008, "slot-EastDrift-Mild",700, 100.0,  0.22, 0.45,  28.79, 16, 4),
    # harder / held-out
    (2001, "slot-EastDrift-Steep", 307, 100.0,  0.75, 0.30, 91.76, 28, 3),
    (2002, "slot-WestDrift-Noisy", 458, 100.0, -0.45, 0.55, 80.12, 22, 3),
]
EFF_HIGH = 1.0
EFF_LOW = 0.2
SAFETY_FRAC = 0.95   # forced-burn margin as a fraction of box_half_width


def make_instances():
    out = []
    for (seed, name, T, W, b, A, F, P, L) in _SPECS:
        public = {"name": name, "horizon": T, "box_half_width": W, "bias": b,
                  "noise_amp": A, "fuel_budget": F, "period": P, "window_len": L,
                  "eff_high": EFF_HIGH, "eff_low": EFF_LOW}
        hidden = {"seed": seed}
        out.append({"public": public, "hidden": hidden})
    return out


# ----------------------------- simulation -----------------------------
def _simulate(pub, seed, policy):
    T = pub["horizon"]; W = pub["box_half_width"]; b = pub["bias"]; A = pub["noise_amp"]
    F = pub["fuel_budget"]; P = pub["period"]; L = pub["window_len"]
    eh = pub["eff_high"]; el = pub["eff_low"]
    u = _rng(seed * 2654435761 + 12345)

    band_lo = policy["band_lo"]; band_hi = policy["band_hi"]
    target = policy["target_pos"]; patience = policy["patience"]

    x = 0.0
    fuel = F
    wait = -1
    for t in range(T):
        noise = u(-A, A)
        x += b + noise
        if x > W or x < -W:
            return t
        eff = eh if (t % P) < L else el
        viol = (x > band_hi) or (x < -band_lo)
        if viol:
            wait = 0 if wait < 0 else wait + 1
        else:
            wait = -1
        urgent = (x > SAFETY_FRAC * W) or (x < -SAFETY_FRAC * W)
        # The 0.95*W safety burn only helps once you are ALREADY outside your own
        # dead-band (viol) -- a band wider than 0.95*W never registers a violation
        # before the hard box-exit at box_half_width, so it gets no protection at all.
        # Keeping band_lo/band_hi inside the safety margin is what buys the safety net.
        do_burn = viol and fuel > 1e-9 and (eff >= eh - 1e-9 or wait >= patience or urgent)
        if do_burn:
            delta = x - target
            cost = abs(delta) / eff
            if cost <= fuel:
                x = target
                fuel -= cost
            else:
                frac = fuel / cost if cost > 0 else 0.0
                x = x - delta * frac
                fuel = 0.0
            wait = -1
    return T


_DO_NOTHING = {"band_lo": 1e12, "band_hi": 1e12, "target_pos": 0.0, "patience": 0}


def baseline(inst):
    # band_lo/band_hi are set beyond box_half_width, so `viol` can never become true
    # before the box-exit check fires -- no burn (not even the safety-net one) ever
    # triggers. This is pure, completely uncontrolled drift: the evaluator's own weak
    # reference.
    return _simulate(inst["public"], inst["hidden"]["seed"], _DO_NOTHING)


def _validate(answer):
    if not isinstance(answer, dict):
        return None
    req = ("band_lo", "band_hi", "target_pos", "patience")
    for k in req:
        if k not in answer:
            return None
    band_lo = answer["band_lo"]; band_hi = answer["band_hi"]
    target = answer["target_pos"]; patience = answer["patience"]
    for v in (band_lo, band_hi, target, patience):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        if v != v or v in (float("inf"), float("-inf")):
            return None
    if band_lo < 0 or band_hi < 0:
        return None
    if abs(target) > 1e6:
        return None
    if patience < 0 or patience > 10 ** 6:
        return None
    return {"band_lo": float(band_lo), "band_hi": float(band_hi),
            "target_pos": float(target), "patience": int(round(patience))}


def score(inst, answer):
    policy = _validate(answer)
    if policy is None:
        return False, 0.0
    pub = inst["public"]; seed = inst["hidden"]["seed"]
    life = _simulate(pub, seed, policy)
    return True, float(life)


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = make_instances()

    vec = []
    for inst in instances:
        pub = inst["public"]
        base = baseline(inst)
        ub = pub["horizon"]
        denom = max(1e-9, ub - base)
        ans, st = isorun.run_candidate(cand, pub, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, life = score(inst, ans)
        except Exception:
            ok, life = False, 0.0
        if not ok:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (life - base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
