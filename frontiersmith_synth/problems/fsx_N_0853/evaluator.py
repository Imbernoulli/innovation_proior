#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_N_0853 -- "Rationed Sprays, Standing Guard"
(family: resistance-sparing-control; format B, quality-metric).

THEME. A season-long integrated-pest-management plan for one field. Each of
T days the field holds a pest population P (crop-eating insects) and a
beneficial-predator population Q (their natural enemy). Once, up front, the
grower must submit a FULL season plan: for every day t a pesticide dose
spray[t] in [0,1] and a predator release amount release[t] in [0,
release_cap]. There is no feedback loop -- the plan is committed in one shot
and then replayed by the evaluator against the same coupled dynamics the
grower could have simulated themselves (every constant needed is in the
public instance).

TWO COMPOSED MECHANISMS drive the objective:

 (1) coupled-predator-prey-dynamics. Pest population follows logistic growth
     minus predation minus chemical kill; predator population grows from
     converted predation minus natural death minus chemical kill plus
     releases:
         P_{t+1} = max(0, P_t + r*P_t*(1-P_t/K) - a*P_t*Q_t - eff_t*P_t)
         Q_{t+1} = max(0, Q_t*(1-d) + b*a*P_t*Q_t - pk_t*Q_t + release_t)
     If Q_t is held near the CRITICAL DENSITY q* = r/a, the predation term
     a*P*q* cancels the logistic growth term at low P, so the ecosystem
     suppresses the pest on its own -- no chemical required.

 (2) intervention-resistance-buildup. A per-instance resistance level R_t
     starts at a small constant (0.02) and ratchets up, NEVER decreasing,
     every time pesticide is used:
         R_{t+1} = min(1, R_t + gamma*spray_t*(1-R_t))
     Pesticide effectiveness on pests decays with resistance (eff_t =
     spray_eff0*spray_t*(1-R_t)); pesticide kill of predators does NOT decay
     with resistance (pk_t = pred_kill0*spray_t), since resistance is
     pest-specific -- so repeated spraying is doubly self-defeating: it gets
     weaker against the pest AND keeps killing the ally that would otherwise
     work for free.

 THE TRAP. The obvious first policy -- "spray at full dose whenever P_t
 exceeds the damage threshold, otherwise do nothing" -- never releases
 predators and reflexively re-sprays every time the pest rebounds. On
 instances where the field's spray_eff0/pred_kill0/gamma mix punishes
 repeated dosing (a "trap" instance: chemical control is fragile), this
 grinds the predator population to ~0 and drives resistance toward 1,
 entering a death spiral where damage recurs every cycle and each spray
 buys less and less. The insight the strong policy exploits: read gamma,
 pred_kill0, spray_eff0, r, a from the instance, decide (per-instance)
 whether a short calibrated opening spray is even worth its resistance/
 predator cost, then invest in reaching the critical predator density q* =
 r/a via releases and hold it there -- ration chemical use to the rare case
 the pest is genuinely running away, so the coupled dynamics do the season's
 remaining suppression for free.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
    {"name": str, "T": int, "P0": float, "Q0": float,
     "r_pest": float, "K_pest": float, "attack_rate": float,
     "convert_eff": float, "pred_death": float, "resist_gain": float,
     "spray_eff0": float, "pred_kill0": float, "loss_coeff": float,
     "damage_thresh": float, "spray_cost": float, "release_cost": float,
     "release_cap": float}
  stdout: ONE JSON object:
    {"spray": [s_0, ..., s_{T-1}], "release": [u_0, ..., u_{T-1}]}
  s_t must be a finite number in [0,1]; u_t a finite number in [0,
  release_cap]. Wrong length/type, out-of-range values, non-finite values,
  a crash, a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time). The evaluator REPLAYS the submitted
plan against the TRUE coupled dynamics (starting from R_0 = 0.02) to get the
season cost:
    cost = sum_t [ loss_coeff*max(0, P_t - damage_thresh)
                    + spray_cost*spray_t + release_cost*release_t ]
(loss uses the pre-action P_t, i.e. damage accrued that day before the
day's intervention takes effect). Reference:
    base = cost of the "never spray, never release" plan on this instance
    r = clamp(0.1 + 0.9 * (base - cost) / base, 0, 1)
Doing nothing exactly reproduces `base`, anchoring r = 0.1. `base` is a
per-instance quantity the evaluator computes itself; it is NOT sent to the
candidate. cost >= 0 always, so r cannot reach 1.0 in practice -- headroom
remains above any reference solution.

ISOLATION. The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the public instance. The season's
true resistance trajectory and the "do nothing" reference cost are computed
only in this parent process.

CLI: python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json
import isorun


# ----------------------------- instance family ------------------------------
# Each row is a fully-specified, hand-calibrated field. Six are engineered
# "trap" instances (fragile chemical control: high resist_gain and/or high
# pred_kill0 relative to spray_eff0) where reflexive spraying is
# self-defeating; four are "benign" instances where a short calibrated
# opening spray is genuinely cheap and worth it before handing control to
# the predators. seed/name only label the instance -- all dynamics are
# deterministic given the plan, so no RNG is needed for reproducibility.
_SPECS = [
    # name,        T,  P0,  Q0, r_pest, K_pest,  attack,  convert, pred_death, resist_gain, spray_eff0, pred_kill0, loss_coeff, damage_thresh, spray_cost, release_cost, release_cap
    ("field-01", 45, 400.0, 15.0, 0.22, 1200.0, 0.0012, 0.35, 0.08, 0.15, 0.75, 0.55, 2.0, 180.0, 90.0, 1.2, 30.0),   # trap
    ("field-02", 45, 500.0, 10.0, 0.28, 1400.0, 0.0010, 0.30, 0.10, 0.20, 0.70, 0.60, 1.8, 200.0, 80.0, 1.0, 25.0),   # trap
    ("field-03", 45, 350.0, 20.0, 0.18, 1000.0, 0.0014, 0.40, 0.07, 0.10, 0.80, 0.45, 2.2, 150.0, 100.0, 1.5, 35.0),  # benign
    ("field-04", 45, 450.0,  8.0, 0.25, 1300.0, 0.0011, 0.32, 0.09, 0.25, 0.65, 0.65, 2.0, 190.0, 75.0, 1.1, 20.0),   # trap
    ("field-05", 45, 300.0, 25.0, 0.15,  900.0, 0.0018, 0.45, 0.06, 0.08, 0.85, 0.35, 2.5, 130.0, 110.0, 1.8, 40.0),  # benign
    ("field-06", 50, 550.0, 12.0, 0.30, 1500.0, 0.0009, 0.28, 0.11, 0.22, 0.70, 0.58, 1.7, 220.0, 70.0, 0.9, 22.0),   # trap
    ("field-07", 40, 380.0, 18.0, 0.20, 1100.0, 0.0013, 0.38, 0.08, 0.12, 0.78, 0.40, 2.1, 160.0, 95.0, 1.4, 32.0),   # benign (held-out shape)
    ("field-08", 45, 420.0,  6.0, 0.24, 1250.0, 0.0010, 0.30, 0.10, 0.30, 0.60, 0.70, 1.9, 175.0, 65.0, 1.0, 18.0),   # trap (severe resistance)
    ("field-09", 45, 320.0, 22.0, 0.16,  950.0, 0.0016, 0.42, 0.07, 0.09, 0.82, 0.38, 2.3, 140.0, 105.0, 1.6, 38.0),  # benign
    ("field-10", 55, 480.0,  9.0, 0.27, 1350.0, 0.0011, 0.31, 0.09, 0.24, 0.68, 0.62, 1.85, 195.0, 78.0, 1.15, 24.0), # trap (long horizon)
]

R0 = 0.02  # fixed background resistance level, same for every instance


def _build_instances():
    out = []
    for (name, T, P0, Q0, r_pest, K_pest, attack, convert, pred_death,
         resist_gain, spray_eff0, pred_kill0, loss_coeff, damage_thresh,
         spray_cost, release_cost, release_cap) in _SPECS:
        out.append(dict(
            name=name, T=T, P0=P0, Q0=Q0, r_pest=r_pest, K_pest=K_pest,
            attack_rate=attack, convert_eff=convert, pred_death=pred_death,
            resist_gain=resist_gain, spray_eff0=spray_eff0, pred_kill0=pred_kill0,
            loss_coeff=loss_coeff, damage_thresh=damage_thresh,
            spray_cost=spray_cost, release_cost=release_cost, release_cap=release_cap,
        ))
    return out


def _public_view(inst):
    keys = ("name", "T", "P0", "Q0", "r_pest", "K_pest", "attack_rate",
            "convert_eff", "pred_death", "resist_gain", "spray_eff0",
            "pred_kill0", "loss_coeff", "damage_thresh", "spray_cost",
            "release_cost", "release_cap")
    return {k: inst[k] for k in keys}


# ----------------------------- shared simulator ------------------------------
def _simulate(inst, spray, release):
    """Replay a full (spray, release) plan through the TRUE dynamics.
    Returns total season cost (float)."""
    T = inst["T"]
    P, Q, R = inst["P0"], inst["Q0"], R0
    r_pest, K_pest = inst["r_pest"], inst["K_pest"]
    attack, convert, pred_death = inst["attack_rate"], inst["convert_eff"], inst["pred_death"]
    resist_gain = inst["resist_gain"]
    spray_eff0, pred_kill0 = inst["spray_eff0"], inst["pred_kill0"]
    loss_coeff, damage_thresh = inst["loss_coeff"], inst["damage_thresh"]
    spray_cost, release_cost = inst["spray_cost"], inst["release_cost"]

    total = 0.0
    for t in range(T):
        s = spray[t]
        u = release[t]
        total += loss_coeff * max(0.0, P - damage_thresh)
        total += spray_cost * s + release_cost * u

        kill_pest = spray_eff0 * s * (1.0 - R)
        kill_pred = pred_kill0 * s
        P_next = max(0.0, P + r_pest * P * (1.0 - P / K_pest) - attack * P * Q - kill_pest * P)
        Q_next = max(0.0, Q * (1.0 - pred_death) + convert * attack * P * Q - kill_pred * Q + u)
        R_next = min(1.0, R + resist_gain * s * (1.0 - R))
        P, Q, R = P_next, Q_next, R_next
    return total


def baseline(inst):
    """The 'do nothing' plan: never spray, never release. The weak (0.1) anchor."""
    T = inst["T"]
    return _simulate(inst, [0.0] * T, [0.0] * T)


# ----------------------------- validation + true replay ---------------------
def score(inst, answer):
    """Validate `answer` and replay it against the TRUE dynamics.
    Returns (ok: bool, cost: float)."""
    if not isinstance(answer, dict):
        return False, 0.0
    spray = answer.get("spray")
    release = answer.get("release")
    T = inst["T"]
    release_cap = inst["release_cap"]

    if not isinstance(spray, list) or len(spray) != T:
        return False, 0.0
    if not isinstance(release, list) or len(release) != T:
        return False, 0.0

    def _finite_num(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and x not in (float("inf"), float("-inf"))

    for s in spray:
        if not _finite_num(s) or s < -1e-9 or s > 1.0 + 1e-9:
            return False, 0.0
    for u in release:
        if not _finite_num(u) or u < -1e-9 or u > release_cap + 1e-9:
            return False, 0.0

    spray_c = [max(0.0, min(1.0, float(s))) for s in spray]
    release_c = [max(0.0, min(release_cap, float(u))) for u in release]

    cost = _simulate(inst, spray_c, release_c)
    if not (cost == cost) or cost in (float("inf"), float("-inf")):
        return False, 0.0
    return True, cost


# ----------------------------- scoring driver --------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = _public_view(inst)
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, cost = score(inst, ans)
        except Exception:
            ok, cost = False, 0.0
        if not ok:
            vec.append(0.0)
            continue
        base = baseline(inst)
        denom = base if base > 1e-9 else 1e-9
        r = 0.1 + 0.9 * (base - cost) / denom
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
